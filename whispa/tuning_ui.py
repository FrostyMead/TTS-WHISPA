"""The "Use your recordings" card on the Voice training page: tune, train, overnight."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QMessageBox, QPushButton, QVBoxLayout, QWidget

from .tuning import RECOGNITION_GOOD, RECOGNITION_MIN, VOICE_GOOD, VOICE_MIN
from .widgets import Label, Switch, card, setting_row


class TuneSection(QWidget):
    def __init__(self, ctl):
        super().__init__()
        self.ctl = ctl
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        head = Label("Use your recordings")
        head.setObjectName("h2")
        outer.addSpacing(10)
        outer.addWidget(head)
        frame, lay = card()
        outer.addWidget(frame)

        self.rec_btn = QPushButton("Tune now")
        self.rec_btn.setObjectName("primary")
        self.rec_btn.clicked.connect(lambda: self._clicked("recognition"))
        self.rec_row = setting_row("Tune speech recognition to my voice", "", self.rec_btn, first=True)
        self.rec_desc = self.rec_row.findChildren(Label)[0] if self.rec_row.findChildren(Label) else None
        self.rec_status = Label("")
        self.rec_status.setObjectName("faint")
        self.rec_status.setWordWrap(True)
        self.rec_row.layout().itemAt(0).widget().layout().addWidget(self.rec_status)
        lay.addWidget(self.rec_row)

        self.ready_btn = QPushButton("Use it")
        self.ready_btn.setObjectName("primary")
        self.ready_btn.clicked.connect(self._use_tuned)
        self.ready_row = setting_row("Your tuned model is ready", "", self.ready_btn)
        self.ready_desc = self.ready_row.findChildren(Label)
        lay.addWidget(self.ready_row)

        self.voice_btn = QPushButton("Train a round")
        self.voice_btn.clicked.connect(lambda: self._clicked("voice"))
        self.voice_row = setting_row("Train my read-aloud voice", "", self.voice_btn)
        self.voice_status = Label("")
        self.voice_status.setObjectName("faint")
        self.voice_status.setWordWrap(True)
        self.voice_row.layout().itemAt(0).widget().layout().addWidget(self.voice_status)
        lay.addWidget(self.voice_row)

        self.auto = Switch(ctl.cfg.auto_tune)
        self.auto.toggled.connect(lambda on: ctl.update_setting("auto_tune", on))
        lay.addWidget(setting_row("Train overnight automatically",
                                  "Between 1 and 6 am, when you haven't used the PC for 30 minutes "
                                  "and there are enough new recordings. Leave the PC on.", self.auto))
        self.blocker = Label("")
        self.blocker.setObjectName("error")
        self.blocker.setWordWrap(True)
        self.blocker.setContentsMargins(0, 4, 0, 10)
        lay.addWidget(self.blocker)

        self._timer = QTimer(self)
        self._timer.setInterval(1500)
        self._timer.timeout.connect(self.refresh)
        self._tuned_path = ""

    @staticmethod
    def _desc(row) -> Label | None:
        labels = [w for w in row.findChildren(Label) if w.objectName() == "muted"]
        return labels[0] if labels else None

    def refresh(self) -> None:
        ctl = self.ctl
        mins = ctl.recorded_minutes()
        blocker = ctl.tuning_blocker()
        self.blocker.setText(blocker)
        self.blocker.setVisible(bool(blocker))
        busy = ctl.tuning_active()

        rec = mins["recognition"]
        job = ctl.tuning_job("recognition")
        desc = (f"{rec:.0f} of {RECOGNITION_GOOD:.0f} minutes recorded (reading and dictations you "
                "corrected). Takes about 30-60 minutes on your graphics card; you can keep "
                "dictating meanwhile, a little slower.")
        if rec < RECOGNITION_MIN:
            desc += f" Record at least {RECOGNITION_MIN:.0f} minutes first."
        elif rec < RECOGNITION_GOOD:
            desc += " Works best from 30 minutes."
        self._set_desc(self.rec_row, desc)
        running = bool(job and job.running)
        self.rec_btn.setText("Cancel" if running else "Tune now")
        self.rec_btn.setEnabled(not blocker and (running or (not busy and rec >= RECOGNITION_MIN)))
        self.rec_status.setText(self._job_text(job))

        # A tuned model that's installed but not in use
        tuned = self._newest_tuned()
        self.ready_row.setVisible(bool(tuned))
        if tuned:
            self._tuned_path = str(tuned.path)
            self._set_desc(self.ready_row, tuned.summary())

        cfg = ctl.cfg
        self.voice_row.setVisible(bool(cfg.voice_clone))
        v = mins["voice"]
        vjob = ctl.tuning_job("voice")
        vdesc = (f"{v:.0f} of {VOICE_GOOD:.0f} minutes of high-quality reading. Each round trains "
                 "for up to 4 hours and continues where the last one stopped; the voice gets "
                 "closer to yours every round. The first round also sets up its tools (about 4 GB).")
        if v < VOICE_MIN:
            vdesc += f" Read at least {VOICE_MIN:.0f} minutes first."
        self._set_desc(self.voice_row, vdesc)
        vrunning = bool(vjob and vjob.running)
        self.voice_btn.setText("Cancel" if vrunning else "Train a round")
        self.voice_btn.setEnabled(not blocker and (vrunning or (not busy and v >= VOICE_MIN)))
        self.voice_status.setText(self._job_text(vjob))
        if busy:
            self._timer.start()
        else:
            self._timer.stop()

    def _set_desc(self, row, text: str) -> None:
        d = self._desc(row)
        if d is None:
            d = Label(text)
            d.setObjectName("muted")
            d.setWordWrap(True)
            row.layout().itemAt(0).widget().layout().insertWidget(1, d)
        d.setText(text)

    @staticmethod
    def _job_text(job) -> str:
        if job is None:
            return ""
        mins = job.elapsed / 60
        if job.running:
            return f"{job.status} · {mins:.0f} min so far"
        return job.status

    def _newest_tuned(self):
        from .speech_model import installed_models, same_path

        models = [m for m in installed_models() if m.recommended]
        if not models or any(same_path(self.ctl.cfg.model, m.path) for m in models[:1]):
            return None
        return models[0]

    def _clicked(self, kind: str) -> None:
        job = self.ctl.tuning_job(kind)
        if job and job.running:
            if QMessageBox.question(self, "Stop training", "Stop this training run?") \
                    == QMessageBox.StandardButton.Yes:
                self.ctl.cancel_tuning(kind)
            return
        err = self.ctl.start_tuning(kind)
        if err:
            QMessageBox.information(self, "Murmr", err)
        self.refresh()

    def _use_tuned(self) -> None:
        if self._tuned_path and QMessageBox.question(
                self, "Use your tuned model", "Switch to the speech model tuned to your voice? "
                "Murmr restarts to load it (a few seconds). You can switch back in Settings > "
                "Speech model.") == QMessageBox.StandardButton.Yes:
            self.ctl.use_tuned_model(self._tuned_path)
