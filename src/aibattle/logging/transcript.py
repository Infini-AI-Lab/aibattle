"""Human-readable plain-text transcript for a single episode trajectory.

Renders what each agent saw, what it output (raw model text for model agents),
and which action was extracted — for easy reading and debugging, as opposed to
the machine-oriented JSON in trajectories.json / match.jsonl.
"""

from __future__ import annotations


def render_transcript(traj: dict) -> str:
    lines = []
    sep = "=" * 70
    lines.append(sep)
    lines.append(f"Episode {traj['episode']}  (pair {traj['pair_id']})")
    lines.append(f"Game:  {traj['game']} v{traj['game_version']}")
    lines.append(f"Seed:  {traj['seed']}")
    seats = ", ".join(f"{p}={n}" for p, n in traj["seat_assignment"].items())
    lines.append(f"Seats: {seats}")
    lines.append(sep)

    for s in traj["steps"]:
        obs = s["observation"]
        resp = s["response"]
        lines.append("")
        lines.append(
            f"--- Step {s['step']} | {s['player']} ({s.get('agent_name', '?')}) ---"
        )
        lines.append(f"Private:       {obs.get('private')}")
        lines.append(f"Public:        {obs.get('public')}")
        lines.append(f"Legal actions: {', '.join(obs.get('legal_actions', []))}")
        lines.append("Observation shown to agent:")
        for ln in (obs.get("rendered") or "").splitlines() or [""]:
            lines.append(f"    {ln}")

        raw = resp.get("raw_output")
        msg = resp.get("message")
        if raw is not None:
            lines.append("Model raw output:")
            for ln in str(raw).splitlines() or [""]:
                lines.append(f"    {ln}")
        elif msg is not None:
            lines.append(f"Agent message: {msg}")

        amt = s.get("selected_amount")
        lines.append(f"Extracted action: {s['selected_action']}"
                     + (f" {amt}" if amt is not None else ""))
        if s.get("invalid"):
            info = s.get("invalid_info", {})
            lines.append(
                f"  [INVALID: requested={info.get('requested')!r} "
                f"reason={info.get('reason')} -> {info.get('resolution')}]"
            )
        meta = resp.get("metadata") or {}
        if meta:
            lines.append(f"  (metadata: {meta})")

    lines.append("")
    lines.append("-" * 70)
    ret = ", ".join(f"{p}={v:+g}" for p, v in traj["returns"].items())
    lines.append(f"Returns: {ret}")
    lines.append(
        f"Winner:  {traj.get('winner')} "
        f"({traj.get('winner_name')})" if traj.get("winner") else "Winner:  tie"
    )
    lines.append(f"Length:  {traj['length']} steps"
                 + ("  [FORFEIT]" if traj.get("forfeit") else ""))
    lines.append(sep)
    lines.append("")
    return "\n".join(lines)
