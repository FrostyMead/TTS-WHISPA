from whispa.watchdog import Watchdog


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def make():
    clock, events = Clock(), []
    w = Watchdog(lambda lag: events.append(("freeze", lag)), lambda lag: events.append(("restart", lag)),
                 dump_after=20, restart_after=85, check_every=2, clock=clock)
    return w, clock, events


def test_healthy_app_is_left_alone():
    w, clock, events = make()
    for _ in range(100):
        clock.t += 2
        w.beat()
        assert w.check(slept=2) == "ok"
    assert events == []


def test_freeze_records_once_then_restarts():
    w, clock, events = make()
    clock.t = 25
    assert w.check(slept=2) == "frozen"
    clock.t = 40
    assert w.check(slept=2) == "frozen"
    assert events == [("freeze", 25)]  # stacks recorded once
    clock.t = 90
    assert w.check(slept=2) == "restart"
    assert events[-1] == ("restart", 90)


def test_computer_sleep_is_not_a_freeze():
    w, clock, events = make()
    clock.t = 3600  # an hour asleep: the checker's own 2 s nap took an hour
    assert w.check(slept=3600) == "ok"
    clock.t = 3602
    assert w.check(slept=2) == "ok" and events == []


def test_recovers_after_a_hiccup():
    w, clock, events = make()
    clock.t = 30
    assert w.check(slept=2) == "frozen"
    w.beat()
    assert w.check(slept=2) == "ok"
    clock.t = 60
    assert w.check(slept=2) == "frozen" and len(events) == 2  # a new freeze is recorded again
