import pytest
import torch

import pytorch_kinematics as pk


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_slerp_mixed_coincident_and_rotating_rows(dtype):
    angles = torch.tensor([0.0, 0.7, 2.0], dtype=dtype)
    zeros = torch.zeros_like(angles)
    q1 = torch.stack((torch.ones_like(angles), zeros, zeros, zeros), -1)
    q2 = torch.stack(((angles / 2).cos(), zeros, zeros, (angles / 2).sin()), -1)
    actual = pk.quaternion_slerp(q1, q2, 0.3)
    expected = torch.stack(((0.3 * angles / 2).cos(), zeros, zeros,
                            (0.3 * angles / 2).sin()), -1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.norm(dim=-1), torch.ones_like(angles))


def test_slerp_does_not_modify_inputs_and_supports_gradients():
    q1 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    q2 = torch.tensor([[-0.8, 0.0, 0.0, -0.6]], dtype=torch.float64, requires_grad=True)
    before = q2.detach().clone()
    result = pk.quaternion_slerp(q1, q2, 0.4)
    torch.testing.assert_close(q2.detach(), before)
    gradients = torch.autograd.grad(result.sum(), (q1, q2))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_slerp_single_quaternion_and_antipodal_identity(sign):
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, requires_grad=True)
    q2 = sign * q1.detach()
    result = pk.quaternion_slerp(q1, q2, 0.5)
    torch.testing.assert_close(result, q1)
    assert torch.isfinite(torch.autograd.grad(result.sum(), q1)[0]).all()


def test_slerp_tensor_parameter_and_broadcast_batch():
    angles = torch.tensor([[0.0, 0.8, 1.7]], dtype=torch.float64)
    zeros = torch.zeros_like(angles)
    q1 = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    q2 = torch.stack(((angles / 2).cos(), zeros, zeros, (angles / 2).sin()), -1)
    t = torch.tensor([[0.0], [0.4], [1.0]], dtype=torch.float64, requires_grad=True)
    actual = pk.quaternion_slerp(q1, q2, t)
    phase = t * angles / 2
    zeros = torch.zeros_like(phase)
    expected = torch.stack((phase.cos(), zeros, zeros, phase.sin()), -1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual[..., -1].sum(), t)[0],
                               (angles / 2 * phase.cos()).sum(-1, keepdim=True))
