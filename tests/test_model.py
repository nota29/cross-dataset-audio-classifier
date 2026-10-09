import pytest
import torch

from src.losses import compute_combined_loss, compute_task_loss, ntxent_loss
from src.model import DualHeadClassifier, ResidualBlock

BATCH, INPUT_DIM, HIDDEN_DIM = 4, 1536, 64
NUM_B, NUM_F = 23, 200


@pytest.fixture
def model():
    return DualHeadClassifier(
        input_dim=INPUT_DIM,
        num_classes_b=NUM_B,
        num_classes_f=NUM_F,
        hidden_dim=HIDDEN_DIM,
        depth=2,
    )


def test_residual_block_keeps_shape():
    block = ResidualBlock(HIDDEN_DIM)
    x = torch.randn(BATCH, HIDDEN_DIM)
    assert block(x).shape == x.shape


@pytest.mark.parametrize("task, num_classes", [("task_b", NUM_B), ("task_f", NUM_F)])
def test_forward_output_shapes(model, task, num_classes):
    logits, features = model(torch.randn(BATCH, INPUT_DIM), task)
    assert logits.shape == (BATCH, num_classes)
    assert features.shape == (BATCH, HIDDEN_DIM)


def test_task_loss_single_label():
    logits = torch.randn(BATCH, NUM_B)
    targets = torch.randint(0, NUM_B, (BATCH,))
    loss = compute_task_loss("task_b", logits, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_task_loss_multi_label():
    logits = torch.randn(BATCH, NUM_F)
    targets = torch.randint(0, 2, (BATCH, NUM_F)).float()
    loss = compute_task_loss("task_f", logits, targets)
    assert loss.dim() == 0 and torch.isfinite(loss)


def test_task_loss_rejects_unknown_task():
    with pytest.raises(ValueError):
        compute_task_loss("task_x", torch.randn(2, 3), torch.zeros(2))


def test_ntxent_is_lower_for_aligned_features():
    torch.manual_seed(0)
    z = torch.randn(8, 32)
    aligned = ntxent_loss(z, z.clone())
    unrelated = ntxent_loss(z, torch.randn(8, 32))
    assert torch.isfinite(aligned) and torch.isfinite(unrelated)
    assert aligned < unrelated


def test_combined_loss_backpropagates(model):
    logits_b, feats_b = model(torch.randn(BATCH, INPUT_DIM), "task_b")
    logits_f, feats_f = model(torch.randn(BATCH, INPUT_DIM), "task_f")
    task_outputs = {
        "task_b": (logits_b, feats_b, torch.randint(0, NUM_B, (BATCH,))),
        "task_f": (logits_f, feats_f, torch.randint(0, 2, (BATCH, NUM_F)).float()),
    }
    loss = compute_combined_loss(
        task_outputs, {"task_b": 0.15, "task_f": 1.0}, alpha_align=0.0001
    )
    assert loss.dim() == 0 and torch.isfinite(loss)

    loss.backward()
    assert model.input_layer.weight.grad is not None
