def test_xT_is_standard_gaussian():
    sched = NoiseScheduler(num_timesteps=1000, schedule="linear")

    x_0 = torch.rand(512, 1, 28, 28) * 2 - 1
    t = torch.full((512,), sched.num_timesteps - 1).long()

    x_T, _ = sched.add_noise(x_0, t)

    mean = x_T.mean().item()
    std = x_T.std().item()
    print(f"x_T mean = {mean:.4f}  (expect ~0)")
    print(f"x_T std  = {std:.4f}  (expect ~1)")

    assert abs(mean) < 0.05, f"mean {mean} not near 0"
    assert abs(std - 1.0) < 0.05, f"std {std} not near 1"


def test_buffer_shapes():
    sched = NoiseScheduler(num_timesteps=1000)
    n = sched.num_timesteps
    for name in ["betas", "alphas", "alpha_bars",
                 "sqrt_alpha_bars", "sqrt_one_minus_alpha_bars"]:
        buf = getattr(sched, name)
        assert buf.shape == (n,), f"{name} has wrong shape {buf.shape}"


def test_alpha_bar_monotonic_decreasing():
    sched = NoiseScheduler(num_timesteps=1000, schedule="cosine")
    ab = sched.alpha_bars
    assert torch.all(ab[1:] <= ab[:-1] + 1e-6), "alpha_bar not decreasing"


def test_t0_is_almost_clean():
    sched = NoiseScheduler(num_timesteps=1000)
    x_0 = torch.rand(16, 1, 28, 28) * 2 - 1
    t = torch.zeros(16).long()
    noise = torch.zeros_like(x_0)
    x_t, _ = sched.add_noise(x_0, t, noise=noise)
    assert torch.allclose(x_t, x_0, atol=1e-2)


if __name__ == "__main__":
    test_buffer_shapes()
    test_alpha_bar_monotonic_decreasing()
    test_t0_is_almost_clean()
    test_xT_is_standard_gaussian()
    print("\nAll tests passed.")
