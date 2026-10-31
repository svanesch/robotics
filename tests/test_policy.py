import numpy as np
import pytest

torch = pytest.importorskip("torch")  # policy tests need the torch feature

from robotics.data.randomize import LIFT_TEMPLATES, PLACE_TEMPLATES
from robotics.policy.rollout import parse_prompt
from robotics.policy.tokenizer import UNK, Tokenizer
from robotics.sim.builder import CUBE_COLORS


def test_tokenizer_covers_every_generated_instruction():
    tok = Tokenizer()
    for t in LIFT_TEMPLATES + PLACE_TEMPLATES:
        for c in CUBE_COLORS:
            ids = tok.encode(t.format(c=c, side="left"))
            assert UNK not in ids and ids[0] == 2
    assert UNK in tok.encode("juggle the banana")


@pytest.mark.parametrize("prompt,expected", [
    ("pick up the red block", ("lift", "red")),
    ("put the blue cube on the left", ("place_left", "blue")),
    ("move the green block to the right pad", ("place_right", "green")),
    ("do something", ("lift", None)),
])
def test_parse_prompt(prompt, expected):
    assert parse_prompt(prompt) == expected


def test_model_forward_backward_and_shapes():
    from robotics.policy.model import VLA, VLAConfig

    cfg = VLAConfig(vocab_size=len(Tokenizer()), pretrained=False, chunk=8, d_model=64, n_heads=4, enc_layers=1,
                    dec_layers=1, ff=128)
    m = VLA(cfg)
    B = 3
    out = m(torch.rand(B, 3, 128, 128), torch.rand(B, 3, 128, 128), torch.randn(B, 6),
            torch.from_numpy(np.stack([Tokenizer().encode("pick up the red block")] * B)))
    assert out.shape == (B, 8, 6)
    out.abs().mean().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


@pytest.fixture(scope="module")
def tiny_dataset(tmp_path_factory):
    from robotics.data.collect import collect_shard

    root = tmp_path_factory.mktemp("ds") / "tiny"
    collect_shard("so101", str(root), 8, 2, 128, 7)
    return root


def test_training_batches_contain_no_privileged_state(tiny_dataset):
    from robotics.policy.data import LABEL_KEYS, MODEL_INPUT_KEYS, VLADataset

    ds = VLADataset(tiny_dataset, "train", chunk=16)
    item = ds[0]
    assert set(item) == set(MODEL_INPUT_KEYS) | set(LABEL_KEYS)
    assert not any("environment" in k or "ee_pose" in k or "cube" in k for k in item)
    assert item["wrist"].shape == (128, 128, 3) and item["actions"].shape == (16, 6) and item["state"].shape == (6,)
    # the model signature cannot even accept extra inputs
    import inspect

    from robotics.policy.model import VLA

    assert list(inspect.signature(VLA.forward).parameters) == ["self", "wrist", "external", "state", "tokens"]


def test_dagger_collection_writes_a_loadable_dataset(tmp_path):
    from robotics.data.dagger import collect_dagger
    from robotics.policy.data import VLADataset
    from robotics.policy.model import VLA, VLAConfig
    from robotics.policy.train_utils import save_checkpoint

    cfg = VLAConfig(vocab_size=len(Tokenizer()), pretrained=False, chunk=16, d_model=64, n_heads=4, enc_layers=1,
                    dec_layers=1, ff=128)
    stats = {"state_mean": np.zeros(6), "state_std": np.ones(6), "action_mean": np.zeros(6), "action_std": np.ones(6)}
    ckpt = tmp_path / "tiny.pt"
    save_checkpoint(ckpt, VLA(cfg), {"dataset": "x"}, stats, 0)
    out = tmp_path / "dg"
    s = collect_dagger(str(ckpt), str(out), episodes=3, cells=2, seed=1, handover_range=(20, 40), max_steps=60, device="cpu")
    assert s["episodes"] >= 3 and s["policy_frames"] > 0
    ds = VLADataset(out, "train", chunk=16)
    item = ds[0]
    assert item["actions"].shape == (16, 6) and item["wrist"].shape == (128, 128, 3)


def test_text_free_model_ignores_the_prompt_and_has_no_text_parameters():
    from robotics.policy.model import VLA, VLAConfig

    cfg = VLAConfig(vocab_size=len(Tokenizer()), pretrained=False, chunk=8, d_model=64, n_heads=4, enc_layers=1, dec_layers=1,
                    ff=128, text=False)
    m = VLA(cfg).eval()
    assert not any(n.startswith(("text_", "film", "query_proj", "key_proj", "coord_proj")) for n, _ in m.named_parameters())
    x, s = torch.rand(2, 3, 128, 128), torch.randn(2, 6)
    a = torch.from_numpy(np.stack([Tokenizer().encode("pick up the red block")] * 2))
    b = torch.from_numpy(np.stack([Tokenizer().encode("put the blue block on the left")] * 2))
    with torch.no_grad():
        assert torch.equal(m(x, x, s, a), m(x, x, s, b)) and torch.equal(m(x, x, s, a), m(x, x, s))
    m.train()
    m(x, x, s).mean().backward()


def test_policy_loss_gripper_bce_is_a_binary_decision():
    from robotics.policy.train_utils import policy_loss

    stats = {"action_mean": np.array([0.0] * 6 + [0.5], np.float32), "action_std": np.array([1.0] * 6 + [0.5], np.float32)}
    actions = torch.zeros(2, 4, 7)
    actions[..., -1] = torch.tensor([-1.0, 1.0]).view(2, 1)  # raw gripper 0 (open) and 1 (close)
    pad = torch.zeros(2, 4, dtype=torch.bool)
    good = torch.zeros(2, 4, 7)
    good[..., -1] = torch.tensor([-8.0, 8.0]).view(2, 1)  # confident logits
    undecided = torch.zeros(2, 4, 7)  # logit 0 = 50/50
    assert policy_loss(good, actions, pad, stats, True) < 0.01
    assert policy_loss(undecided, actions, pad, stats, True) > 0.6
    wrong = -good
    assert policy_loss(wrong, actions, pad, stats, True) > 5.0
    assert policy_loss(good, actions, pad, stats, False) >= 0.0  # the L1 path is unchanged
