"""k9s-style TUI for ablator: queue/runs views, config editing, kubeconfig
context switching. Scoped to ablator's own domain only.

Imports `textual` lazily (inside `launch()`), so `import ablator.tui.app`
never fails for a headless install -- only calling `launch()` without the
`ablator[tui]` extra installed does, with a clear message.
"""
from __future__ import annotations

import os

from .. import cli as climod
from .. import config as cfgmod
from . import contexts as ctxmod
from . import pod_status as podmod
from . import queue_view as qvmod
from . import wizard as wizardmod


def launch(config_path: str | None = None) -> None:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Vertical
        from textual.screen import Screen
        from textual.widgets import (DataTable, Footer, Header, Label,
                                     ListItem, ListView, Static)
    except ImportError as e:
        raise SystemExit(
            "ablator: the TUI needs the 'textual' package -- install with "
            "`pip install ablator[tui]`") from e

    resolved_path = (config_path or os.environ.get("ABLATOR_CONFIG")
                    or cfgmod.DEFAULT_CONFIG_PATH)
    if not os.path.exists(resolved_path):
        wizardmod.run_wizard(resolved_path)
    cfg = cfgmod.load_config(resolved_path)

    class QueueScreen(Screen):
        BINDINGS = [("c", "cancel_selected", "Cancel"), ("r", "refresh", "Refresh")]

        def compose(self) -> ComposeResult:
            yield Header()
            yield DataTable(id="queue_table")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#queue_table", DataTable)
            table.add_columns(*qvmod.COLUMNS)
            self.action_refresh()
            self.set_interval(5.0, self.action_refresh)

        def action_refresh(self) -> None:
            table = self.query_one("#queue_table", DataTable)
            table.clear()
            for row in qvmod.queue_rows(qvmod.load_jobs(cfg)):
                table.add_row(*row)

        def action_cancel_selected(self) -> None:
            table = self.query_one("#queue_table", DataTable)
            if table.cursor_row is None:
                return
            row = table.get_row_at(table.cursor_row)
            job_id = row[0]
            try:
                climod.cmd_control(cfg, "skip", job_id)
            except SystemExit as e:
                self.notify(str(e))
            self.action_refresh()

    class RunsScreen(Screen):
        BINDINGS = [("r", "refresh", "Refresh")]

        def compose(self) -> ComposeResult:
            yield Header()
            yield DataTable(id="runs_table")
            yield Static("", id="pod_detail")
            yield Footer()

        def on_mount(self) -> None:
            table = self.query_one("#runs_table", DataTable)
            table.add_columns(*qvmod.COLUMNS)
            self.action_refresh()
            self.set_interval(5.0, self.action_refresh)

        def action_refresh(self) -> None:
            table = self.query_one("#runs_table", DataTable)
            table.clear()
            for row in qvmod.running_rows(qvmod.load_jobs(cfg)):
                table.add_row(*row)

        def on_data_table_row_selected(self, event) -> None:
            row = event.data_table.get_row_at(event.cursor_row)
            job_id, _, _, machine, *_ = row
            mcfg = cfgmod.machine_cfg(cfg, machine)
            detail = self.query_one("#pod_detail", Static)
            job = next((j for j in qvmod.load_jobs(cfg) if j.get("id") == job_id), {})
            source_line = qvmod.source_detail(job)
            if mcfg.get("backend") != "k8s":
                detail.update(f"{source_line}\n{job_id}: bare-metal job on {machine} "
                             "(no pod status to show)")
                return
            k8s_name = qvmod.k8s_job_name(job_id)
            status_line = podmod.pod_status_line(mcfg["namespace"], job_id, k8s_name)
            tail = podmod.recent_log_tail(mcfg["namespace"], job_id, k8s_name)
            detail.update(f"{source_line}\n{job_id} pod: {status_line}\n\n{tail}")

    class ConfigScreen(Screen):
        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(_config_summary(cfg), id="config_summary")
            yield Footer()

    class ContextsScreen(Screen):
        BINDINGS = [("enter", "use_selected", "Use context")]

        def compose(self) -> ComposeResult:
            yield Header()
            yield ListView(id="contexts_list")
            yield Static("", id="context_status")
            yield Footer()

        def on_mount(self) -> None:
            lv = self.query_one("#contexts_list", ListView)
            try:
                names = ctxmod.list_contexts()
                current = ctxmod.current_context()
            except ctxmod.KubectlError as e:
                self.query_one("#context_status", Static).update(str(e))
                return
            for name in names:
                marker = " (current)" if name == current else ""
                lv.append(ListItem(Label(f"{name}{marker}")))

        def action_use_selected(self) -> None:
            lv = self.query_one("#contexts_list", ListView)
            if lv.index is None:
                return
            label = lv.children[lv.index].query_one(Label)
            name = str(label.renderable).replace(" (current)", "")
            status = self.query_one("#context_status", Static)
            try:
                ctxmod.use_context(name)
                status.update(f"switched to {name}")
            except ctxmod.KubectlError as e:
                status.update(f"error: {e}")

    class AblatorTUI(App):
        BINDINGS = [
            ("1", "show_queue", "Queue"),
            ("2", "show_runs", "Runs"),
            ("3", "show_config", "Config"),
            ("4", "show_contexts", "Contexts"),
            ("q", "quit", "Quit"),
        ]

        def on_mount(self) -> None:
            # push_screen, not switch_screen, for the very first screen --
            # switch_screen() pops a result callback off the CURRENT top
            # screen, but at initial mount there's only the App's implicit
            # default screen, which was never push_screen()'d and so has
            # no result callback to pop (IndexError: pop from empty list).
            # Found live via a headless run_test() smoke test.
            self.push_screen(QueueScreen())

        def action_show_queue(self) -> None:
            self.switch_screen(QueueScreen())

        def action_show_runs(self) -> None:
            self.switch_screen(RunsScreen())

        def action_show_config(self) -> None:
            self.switch_screen(ConfigScreen())

        def action_show_contexts(self) -> None:
            self.switch_screen(ContextsScreen())

    AblatorTUI().run()


def _config_summary(cfg: dict) -> str:
    lines = [f"queue.path = {cfg.get('queue', {}).get('path', '')}", "", "machines:"]
    for name, m in cfg.get("machines", {}).items():
        if m.get("backend") == "k8s":
            lines.append(f"  {name}: k8s namespace={m.get('namespace')} "
                        f"queue={m.get('kai_queue')} priority={m.get('priority_class')} "
                        f"image={m.get('image')} gpu_count={m.get('gpu_count', 1)}")
        else:
            lines.append(f"  {name}: bare-metal "
                        f"(patterns={m.get('hostname_patterns', [])})")
    return "\n".join(lines)
