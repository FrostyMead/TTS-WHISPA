from whispa import config as config_mod
from whispa import privacy
from whispa.config import Config


def test_excluded_apps():
    apps = Config().training_excluded_apps
    assert privacy.excluded("1Password", apps)
    assert privacy.excluded("KeePassXC", apps)
    assert privacy.excluded("keychain access", apps)
    assert not privacy.excluded("Slack", apps)
    assert not privacy.excluded("", apps)


def test_voice_is_opt_in_for_new_users():
    cfg = Config()
    assert cfg.save_training_audio is False and cfg.voice_clone is False


def test_delete_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(config_mod, "config_dir", lambda: tmp_path)
    for sub in ("training", "models/tuned-1", "voices/me"):
        (tmp_path / sub).mkdir(parents=True)
    (tmp_path / "training" / "a.wav").write_bytes(b"\0" * 64000)
    (tmp_path / "models" / "tuned-1" / "config.json").write_text("{}")
    st = privacy.storage()
    assert st["clips"] == 1 and st["models"] == 1 and st["voices"] == 1 and round(st["seconds"]) == 2
    cfg = Config()
    cfg.model = str(tmp_path / "models" / "tuned-1")
    monkeypatch.setattr(config_mod, "config_path", lambda: tmp_path / "config.json")
    assert privacy.delete_everything(cfg) is True  # a tuned model was active: restart
    from whispa.asr import BASE_MODEL
    assert cfg.model == BASE_MODEL
    st = privacy.storage()
    assert st["clips"] == 0 and st["models"] == 0 and st["voices"] == 0
