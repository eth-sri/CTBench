"""CLI and training integration tests.

Run from the repository root: python -m unittest Utility.test_training -v

Uses CPU fixtures, mocked teacher checkpoints and CPU to meta transfers.
No datasets, training artifacts or downloaded checkpoints are required.
"""
import ast
import contextlib
import copy
import functools
import io
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
# Load before utils imports pip's vendored typing extensions on newer PyTorch.
try:
    import torch._dynamo
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from args_factory import get_args
import mix_train
from model_wrapper import get_model_wrapper
from AIDomains.abstract_layers import Sequential, _BatchNorm
from AIDomains.concrete_layers import Normalization
from attacks import adv_whitebox
from utils import Scheduler

BASE = '0f18162cba85c54d7fc28af62b2c5926efd3695d'
COMMON = ['--dataset', 'mnist', '--net', 'cnn_7layer_bn', '--train-eps', '.1',
          '--test-eps', '.1', '--train-steps', '1', '--test-steps', '1', '--lr', '.1',
          '--lr-milestones', '1', '3', '--save-dir', 'test-models',
          '--start-value-robust-weight', '1', '--end-value-robust-weight', '1']
X = torch.tensor([[.2, .4], [.4, .8], [.7, .3], [.6, .5]])
Y = torch.tensor([0, 1, 0, 1])


def parse(flags, parser=get_args):
    with patch.object(sys, 'argv', ['probe'] + COMMON + flags):
        return parser(['basic', 'train'])


@functools.lru_cache(None)
def source(filename, ref=None):
    if ref is None:
        return (ROOT / filename).read_text()
    result = subprocess.run(['git', 'show', f'{ref}:{filename}'], cwd=ROOT,
                            capture_output=True, text=True)
    if result.returncode:
        raise unittest.SkipTest(f'Base commit {ref} is unavailable locally.')
    return result.stdout


def definitions(filename, ref, names, namespace):
    nodes = [n for n in ast.parse(source(filename, ref)).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)


@functools.lru_cache(None)
def baseline():
    ns = {'os': os, 'logging': logging}
    exec(compile(source('args_factory.py', BASE), 'base parser', 'exec'), ns)
    definitions('mix_train.py', BASE, ['get_train_mode', 'parse_save_root'], ns)
    return ns


def run_tree(ref=None):
    return next(n for n in ast.parse(source('mix_train.py', ref)).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run')


def epoch_lr_step(ref=None):
    loop = next(n for n in run_tree(ref).body if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == 'epoch_idx')
    nodes = [n for n in loop.body if
             (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
              and isinstance(n.test.left, ast.Attribute) and n.test.left.attr == 'lr_schedule')
             or (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == 'lr_schedular.step')]
    return compile(ast.Module(body=nodes, type_ignores=[]), 'epoch LR update', 'exec')


def distillation_args(method, extra=()):
    return parse([f'--use-{method}-training', '--teacher-model', 'teacher.ckpt',
                  '--model-selection', 'None', '--ibp-coef', '.2'] + list(extra))


def wrapper(args, bn=False):
    layers = [Normalization((2,), mean=[0., 0.], std=[1., 1.]), nn.Linear(2, 3)]
    if bn:
        layers.append(nn.BatchNorm1d(3))
    net = Sequential.from_concrete_network(nn.Sequential(*layers, nn.ReLU(), nn.Linear(3, 3)), (2,))
    net.set_dim(torch.zeros_like(X))
    with patch.object(torch, 'load', return_value=copy.deepcopy(net.state_dict())), \
            contextlib.redirect_stdout(io.StringIO()):
        result = get_model_wrapper(args, net, 'cpu', (2,))
    result.robust_weight = 1.
    return result


def driver_setup(args, model=None):
    """Run actual optimizer/scheduler setup, stopping before training or saving."""
    net = nn.Sequential(nn.Identity(), nn.Linear(2, 3)) if model is None else model.net
    model = SimpleNamespace(net=net) if model is None else model
    loader = type('Loader', (), {'batch_size': 4, '__len__': lambda self: 4})()
    captured = {}

    class SetupComplete(Exception):
        pass

    def stop(*call_args, **kwargs):
        captured['optimizer'], captured['scheduler'] = call_args[5:7]
        raise SetupComplete()

    with contextlib.redirect_stdout(io.StringIO()), \
            patch.object(mix_train, 'get_loaders', return_value=([loader, loader], 2, 1, 3)), \
            patch.object(mix_train, 'get_network', return_value=net), \
            patch.object(Sequential, 'from_concrete_network', return_value=net), \
            patch.object(net, 'set_dim', create=True), \
            patch.object(torch, 'zeros', return_value=None), \
            patch.object(mix_train, 'get_model_wrapper', return_value=model), \
            patch.object(os, 'makedirs'), \
            patch.object(mix_train, 'train_loop', side_effect=stop):
        try:
            mix_train.run(args)
        except SetupComplete:
            pass
    return net, captured['optimizer'], captured['scheduler']


def train_batch(model, optimizer=None, scheduler=None):
    optimizer = torch.optim.SGD(model.net.parameters(), lr=.01) if optimizer is None else optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1]) if scheduler is None else scheduler
    return mix_train.train_loop(model, Scheduler(0, 10, .05, .1), Scheduler(0, 0, 1., 1.),
                                [(X, Y)], 0, optimizer, scheduler, 'cpu', model.args)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        self.addCleanup(logging.disable, previous_logging)
        progress = mix_train.tqdm
        pbar = patch.object(mix_train, 'tqdm', side_effect=lambda data: progress(data, disable=True))
        pbar.start()
        self.addCleanup(pbar.stop)
        autocast = torch.autocast

        def cpu_autocast(*args, **kwargs):
            # Torch 1.12 rejects CPU float16 even when autocast is disabled.
            if kwargs.get('device_type') == 'cpu' and not kwargs.get('enabled', True):
                kwargs['dtype'] = torch.bfloat16
            return autocast(*args, **kwargs)

        amp = patch.object(torch, 'autocast', side_effect=cpu_autocast)
        amp.start()
        self.addCleanup(amp.stop)

    def test_legacy_modes_and_paths_match_base(self):
        old = baseline()
        for method in ['std', 'pgd', 'multipgd', 'arow', 'mart', 'ibp', 'taps', 'DP',
                       'DPBox', 'mtlibp', 'expibp', 'ccibp']:
            with self.subTest(method=method):
                flags = [f'--use-{method}-training']
                if method == 'taps':
                    flags += ['--block-sizes', '10', '2']
                before, after = parse(flags, old['get_args']), parse(flags)
                mode = old['get_train_mode'](before)
                self.assertEqual(mix_train.get_train_mode(after), mode)
                self.assertFalse(after.use_adcert_training or after.use_ccdist_training)
                self.assertIsNone(after.teacher_model)
                with patch.object(os, 'makedirs'):
                    self.assertEqual(old['parse_save_root'](before, mode),
                                     mix_train.parse_save_root(after, mode))

    def test_distillation_modes_and_variants_train(self):
        cases = [('adcert', []), ('ccdist', []), ('adcert', ['--use-rslad']),
                 ('adcert', ['--use-nat-kl']), ('adcert', ['--use-softlabel-attack']),
                 ('adcert', ['--teacher-input', 'adv'])]
        for method, flags in cases:
            with self.subTest(method=method, flags=flags):
                args = distillation_args(method, flags)
                self.assertEqual(mix_train.get_train_mode(args), f'{method.upper()}_trained')
                with patch.object(os, 'makedirs'):
                    mix_train.parse_save_root(args, mix_train.get_train_mode(args))
                model = wrapper(args, bn=True)
                teacher_state = copy.deepcopy(model.teacher.state_dict())
                result = train_batch(model)
                self.assertTrue(torch.isfinite(torch.tensor(result[-1])))
                self.assertTrue(any(p.grad is not None for p in model.net.parameters()))
                self.assertTrue(all(p.grad is None for p in model.teacher.parameters()))
                self.assertTrue(all(not layer.training for layer in model.teacher.modules()
                                    if isinstance(layer, _BatchNorm)))
                for name, value in teacher_state.items():
                    torch.testing.assert_close(value, model.teacher.state_dict()[name])

    def test_legacy_sgd_updates_match_base(self):
        constructor = next(n for n in ast.walk(run_tree(BASE)) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == 'SGD')
        for flags in [[], ['--momentum', '.9'], ['--momentum', '.5'], ['--momentum', '0']]:
            with self.subTest(flags=flags):
                args = parse(['--use-pgd-training', '--opt', 'sgd'] + flags)
                net, optimizer, _ = driver_setup(args)
                old_p = nn.Parameter(torch.tensor(1.))
                old_opt = eval(compile(ast.Expression(constructor), 'base SGD', 'eval'),
                               {'torch': torch, 'args': args, 'param_list': [old_p], 'lr': args.lr})
                parameter = next(net[1].parameters())
                with torch.no_grad():
                    parameter.fill_(1.)
                for _ in range(2):
                    parameter.grad, old_p.grad = torch.ones_like(parameter), torch.ones_like(old_p)
                    optimizer.step()
                    old_opt.step()
                torch.testing.assert_close(parameter, torch.full_like(parameter, old_p.item()))
                self.assertAlmostEqual(old_p.item(), .8, places=6)

    def test_sgd_momentum_opt_in(self):
        for momentum, expected in [(.9, .71), (.5, .75)]:
            with self.subTest(momentum=momentum):
                args = parse(['--use-pgd-training', '--opt', 'sgd', '--use-sgd-momentum',
                              '--momentum', str(momentum)])
                net, optimizer, _ = driver_setup(args)
                parameter = next(net[1].parameters())
                with torch.no_grad():
                    parameter.fill_(1.)
                for _ in range(2):
                    parameter.grad = torch.ones_like(parameter)
                    optimizer.step()
                torch.testing.assert_close(parameter, torch.full_like(parameter, expected))

    def test_multistep_scheduler_matches_base(self):
        args = parse(['--use-std-training'])
        _, optimizer, scheduler = driver_setup(args)
        old_optimizer = torch.optim.Adam([nn.Parameter(torch.tensor(1.))], lr=args.lr)
        old_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            old_optimizer, args.lr_milestones, gamma=args.lr_decay_factor)
        for _ in range(5):
            self.assertEqual(scheduler.get_last_lr(), old_scheduler.get_last_lr())
            optimizer.step()
            old_optimizer.step()
            exec(epoch_lr_step(), {'args': args, 'lr_schedular': scheduler})
            exec(epoch_lr_step(BASE), {'args': args, 'lr_schedular': old_scheduler})

    def test_cyclic_sgd_steps_lr_and_momentum(self):
        args = distillation_args('adcert', ['--opt', 'sgd', '--lr-schedule', 'cyclic',
                                          '--lr-min', '.01', '--n-epochs', '2'])
        model = wrapper(args)
        _, optimizer, scheduler = driver_setup(args, model)
        self.assertTrue(scheduler.cycle_momentum)
        self.assertAlmostEqual(optimizer.param_groups[0]['momentum'], .9)
        train_batch(model, optimizer, scheduler)
        self.assertAlmostEqual(scheduler.get_last_lr()[0], .0325)
        self.assertAlmostEqual(optimizer.param_groups[0]['momentum'], .875)
        self.assertEqual(model.current_lr, scheduler.get_last_lr()[0])

    def test_implementation_specific_option_validation(self):
        invalid = [
            (['--use-pgd-training', '--lr-schedule', 'cyclic'], 'only supports SGD'),
            (['--use-std-training', '--use-sgd-momentum'], 'requires --opt sgd'),
            (['--use-adcert-training'], 'require --teacher-model'),
            (['--use-ccdist-training'], 'require --teacher-model'),
            (['--use-std-training', '--teacher-model', 'teacher.ckpt'], 'Teacher options'),
            (['--use-adcert-training', '--teacher-model', 'teacher.ckpt',
              '--use-rslad', '--use-nat-kl'], 'mutually exclusive'),
            (['--use-adcert-training', '--teacher-model', 'teacher.ckpt'], '--model-selection None'),
            (['--use-adcert-training', '--teacher-model', 'teacher.ckpt',
              '--teacher-input', 'invalid'], 'invalid choice'),
        ]
        for method in ['adcert', 'ccdist']:
            for option in [['--use-EDAC-step'], ['--use-small-box'], ['--relu-shrinkage', '.5']]:
                invalid.append(([f'--use-{method}-training', '--teacher-model', 'teacher.ckpt',
                                 '--model-selection', 'None'] + option, 'do not support'))
        for option in [['--use-rslad'], ['--use-nat-kl'], ['--use-softlabel-attack'],
                       ['--teacher-input', 'adv']]:
            invalid.append((['--use-ccdist-training', '--teacher-model', 'teacher.ckpt',
                             '--model-selection', 'None'] + option, 'require --use-adcert-training'))
        for flags, message in invalid:
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    parse(flags)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())
        for method in ['adcert', 'ccdist']:
            for selection in ['robust_accu', 'loss']:
                wrapper(distillation_args(method, ['--ibp-coef', '0', '--model-selection', selection]))
        for other in ['--use-pgd-training', '--use-ccdist-training']:
            args = distillation_args('adcert', [other])
            with self.assertRaisesRegex(AssertionError, 'Only one training method'):
                mix_train.get_train_mode(args)

    def test_fast_regularization_with_zero_ibp(self):
        for method, coefficient in [('adcert', '0'), ('ccdist', '0'),
                                    ('adcert', '1'), ('ccdist', '1')]:
            with self.subTest(method=method, coefficient=coefficient):
                args = distillation_args(method, ['--ibp-coef', coefficient, '--fast-reg', '.5',
                                                 '--min-eps-reg', '1e-6', '--end-epoch-eps', '2'])
                model = wrapper(args)
                result = train_batch(model)
                self.assertTrue(torch.isfinite(torch.tensor(result[-1])))
                self.assertTrue(any(p.grad is not None for p in model.net.parameters()))

    def test_teacher_transfer_direct_and_nested(self):
        cases = [[], ['--grad-accu-batch', '2'], ['--use-sam'], ['--use-weight-smooth'],
                 ['--grad-accu-batch', '2', '--use-sam'],
                 ['--grad-accu-batch', '2', '--use-weight-smooth']]
        for method in ['adcert', 'ccdist']:
            for flags in cases:
                with self.subTest(method=method, flags=flags):
                    model = wrapper(distillation_args(method, flags), bn=True)
                    chain = [model]
                    while hasattr(chain[-1], 'wrapper'):
                        chain.append(chain[-1].wrapper)
                    inner = chain[-1]
                    self.assertIs(model.to('meta'), model)
                    self.assertTrue(all(str(item.device) == 'meta' for item in chain))
                    names = ['net', 'teacher']
                    if method == 'ccdist':
                        names += ['feature_net', 'head_net', 'teacher_feature_net', 'teacher_head_net']
                    for name in names:
                        module = getattr(inner, name)
                        self.assertTrue(all(t.device.type == 'meta' for t in
                                            list(module.parameters()) + list(module.buffers())))
                    # Torch 1.12 has no meta BatchNorm/ReLU kernels.
                    if int(torch.__version__.split('.')[0]) >= 2:
                        output = inner.teacher(torch.empty_like(X, device='meta'))
                        self.assertEqual(output.device.type, 'meta')

    def test_checkpoint_selection_matches_base(self):
        def decisions(ref, selection, eps, accuracy, loss):
            loop = next(n for n in run_tree(ref).body if isinstance(n, ast.For)
                        and isinstance(n.target, ast.Name) and n.target.id == 'epoch_idx')
            nodes = [n for n in loop.body if isinstance(n, ast.If) and
                     ('eps == args.test_eps' in ast.unparse(n.test)
                      or 'args.save_every_epoch' in ast.unparse(n.test)
                      or 'model_selection' in ast.unparse(n.test))]
            saves = []
            perf = {'model_selection': selection, 'best_val_robust_accu': .6, 'best_val_loss': 1.}
            env = dict(mix_train.__dict__, torch=SimpleNamespace(save=lambda state, path: saves.append(path)),
                       args=SimpleNamespace(test_eps=.1, save_every_epoch=True, n_epochs=2), eps=eps,
                       val_robust_accu=accuracy, val_loss=loss, perf_dict=perf, epoch_idx=1,
                       optimizer=None, model_wrapper=SimpleNamespace(net=nn.Linear(2, 3)),
                       save_root='test-models')
            with patch.object(os, 'makedirs'), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(ast.Module(body=nodes, type_ignores=[]), 'checkpoint decisions', 'exec'), env)
            return saves, perf

        for selection, eps, accuracy, loss in [('robust_accu', .1, .7, 2.),
                                               ('robust_accu', .1, .5, 2.),
                                               ('robust_accu', .05, .7, 2.),
                                               ('loss', .1, .5, .5),
                                               ('loss', .1, .5, 2.), (None, .1, .5, 2.)]:
            with self.subTest(selection=selection, eps=eps, accuracy=accuracy, loss=loss):
                self.assertEqual(decisions(None, selection, eps, accuracy, loss),
                                 decisions(BASE, selection, eps, accuracy, loss))

    def test_ccdist_split_bounds_match_full_network(self):
        model = wrapper(distillation_args('ccdist'))
        model.net.eval()
        lower, upper = X - .1, X + .1
        whole, _ = model.get_IBP_bounds(model.net, lower, upper, Y)
        feature_lower, feature_upper = model.get_IBP_bounds(model.feature_net, lower, upper)
        split, _ = model.get_IBP_bounds(model.head_net, feature_lower, feature_upper, Y)
        torch.testing.assert_close(whole, split)
        torch.testing.assert_close(model.net(X), model.head_net(model.feature_net(X)))

    def test_kl_attack_uses_clean_teacher_once(self):
        model = wrapper(distillation_args('adcert', ['--use-softlabel-attack']))
        with patch.object(model.teacher, 'forward', wraps=model.teacher.forward) as calls:
            adversarial = adv_whitebox(model.net, X, Y, X - .1, X + .1, 'cpu',
                                      num_steps=1, lossFunc='KL', teacher=model.teacher)
        self.assertEqual(calls.call_count, 1)
        torch.testing.assert_close(calls.call_args.args[0], X)
        self.assertFalse(adversarial.requires_grad)
        self.assertTrue(((adversarial >= X - .1) & (adversarial <= X + .1)).all())

    def test_all_ten_example_commands_parse(self):
        paths = sorted((ROOT / 'scripts/examples').glob('*/*/train_adcert.sh'))
        paths += sorted((ROOT / 'scripts/examples').glob('*/*/train_ccdist.sh'))
        self.assertEqual(len(paths), 10)
        # Capture actual Bash argument expansion without launching training.
        capture = 'python() { printf "%s\\0" "$@"; }; python3() { python "$@"; }; source "$1"'
        for path in paths:
            with self.subTest(script=str(path.relative_to(ROOT))):
                subprocess.run(['bash', '-n', str(path)], check=True)
                result = subprocess.run(['bash', '-c', capture, 'capture', str(path)],
                                        cwd=ROOT, capture_output=True, check=True)
                tokens = [value.decode() for value in result.stdout.split(b'\0')[:-1]]
                self.assertEqual(tokens.count('mix_train.py'), 1)
                with patch.object(sys, 'argv', ['probe'] + tokens[tokens.index('mix_train.py') + 1:]):
                    args = get_args(['basic', 'train'])
                expected = 'ADCERT_trained' if path.stem == 'train_adcert' else 'CCDIST_trained'
                self.assertEqual(mix_train.get_train_mode(args), expected)
                with patch.object(os, 'makedirs'):
                    mix_train.parse_save_root(args, expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
