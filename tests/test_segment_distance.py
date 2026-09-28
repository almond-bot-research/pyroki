"""Closest points between segments, the core of every capsule-capsule check."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pyroki.collision import Capsule, collide
from pyroki.collision._utils import closest_segment_to_segment_points


def _brute_force(a1, b1, a2, b2, n=801):
    s = np.linspace(0.0, 1.0, n)[:, None]
    p = a1 + (b1 - a1) * s
    q = a2 + (b2 - a2) * s
    return np.min(np.linalg.norm(p[:, None] - q[None], axis=-1))


def _gap(a1, b1, a2, b2):
    c1, c2 = closest_segment_to_segment_points(
        *(jnp.asarray(x, dtype=jnp.float32) for x in (a1, b1, a2, b2))
    )
    return float(jnp.linalg.norm(c1 - c2))


@pytest.mark.parametrize(
    "a2, b2",
    [
        # Antiparallel, overlapping in height: the regression. The old
        # routine paired a2's projection with a1's and read ~0.42 m.
        ([0.3, 0.0, 0.3], [0.3, 0.0, -0.7]),
        # Parallel, same direction, overlapping.
        ([0.3, 0.0, -0.2], [0.3, 0.0, 0.5]),
        # Parallel, disjoint along the axis.
        ([0.3, 0.0, 0.6], [0.3, 0.0, 0.9]),
        # Collinear, disjoint.
        ([0.0, 0.0, 0.5], [0.0, 0.0, 0.8]),
    ],
)
def test_parallel_segments(a2, b2):
    a1, b1 = np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.34])
    a2, b2 = np.asarray(a2), np.asarray(b2)
    assert _gap(a1, b1, a2, b2) == pytest.approx(_brute_force(a1, b1, a2, b2), abs=1e-4)


def test_random_segments_match_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a1, b1, a2, b2 = rng.normal(size=(4, 3))
        if rng.random() < 0.3:  # parallel / antiparallel
            b2 = a2 + (b1 - a1) * rng.uniform(-2.0, 2.0)
        assert _gap(a1, b1, a2, b2) == pytest.approx(
            _brute_force(a1, b1, a2, b2), abs=2e-3
        )


def test_degenerate_segments_have_finite_gradients():
    def gap(x):
        c1, c2 = closest_segment_to_segment_points(x[0], x[1], x[2], x[3])
        return jnp.sum((c1 - c2) ** 2)

    cases = [
        jnp.array([[0.0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]]),  # parallel
        jnp.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]),  # point
        jnp.array([[0.0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]]),  # two points
    ]
    for x in cases:
        assert np.isfinite(np.asarray(jax.grad(gap)(x))).all()


def test_antiparallel_capsules_distance():
    # Two upright capsules 0.3 m apart (axis to axis), pointing opposite ways.
    def upright(x, flip):
        cap = Capsule.from_radius_height(
            position=jnp.array([x, 0.0, 0.17]),
            wxyz=jnp.array([0.0, 1.0, 0.0, 0.0]) if flip else jnp.array([1.0, 0, 0, 0]),
            radius=jnp.array(0.05),
            height=jnp.array(0.34),
        )
        return cap

    d = float(collide(upright(0.0, False), upright(0.3, True)))
    assert d == pytest.approx(0.3 - 0.1, abs=1e-5)
