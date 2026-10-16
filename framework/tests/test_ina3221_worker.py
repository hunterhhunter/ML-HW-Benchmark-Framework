"""INA3221 register decoding and the Jetson input-power boundary."""

import pytest


class RegisterBus:
    def __init__(self, words):
        self.words = words
        self.writes = []

    def read_word_data(self, address, register):
        assert address == 0x40
        return self.words[register]

    def write_word_data(self, address, register, value):
        self.writes.append((address, register, value))
        self.words[register] = value


def test_reads_positive_load_side_dc_power_from_real_register_layout():
    from monitors.ina3221_worker import read_sample, verify_identity

    # SMBus word reads swap the INA3221 register's high and low bytes.
    bus = RegisterBus({
        0xFE: 0x4954,  # 0x5449
        0xFF: 0x2032,  # 0x3220
        0x01: 0x400B,  # 360 * 40 uV = 14.4 mV across 0.05 ohm
        0x02: 0x004B,  # 2400 * 8 mV = 19.2 V
    })

    verify_identity(bus, 0x40)
    sample = read_sample(bus, 0x40, shunt_ohms=0.05, max_current_a=3.0)

    assert sample["bus_v"] == pytest.approx(19.2)
    assert sample["current_a"] == pytest.approx(0.288)
    assert sample["power_w"] == pytest.approx(5.5296)


def test_rejects_wrong_sensor_and_negative_or_over_limit_current():
    from monitors.ina3221_worker import read_sample, verify_identity

    with pytest.raises(ValueError, match="sensor_id"):
        verify_identity(RegisterBus({0xFE: 0, 0xFF: 0}), 0x40)

    negative = RegisterBus({0x01: 0xC0FF, 0x02: 0x004B})
    with pytest.raises(ValueError, match="negative_current"):
        read_sample(negative, 0x40, shunt_ohms=0.05, max_current_a=3.0)

    over_limit = RegisterBus({0x01: 0x1078, 0x02: 0x004B})
    with pytest.raises(ValueError, match="overcurrent"):
        read_sample(over_limit, 0x40, shunt_ohms=0.05, max_current_a=3.0)


def test_configures_only_channel_one_for_continuous_shunt_and_bus_conversion():
    from monitors.ina3221_worker import configure_sensor

    bus = RegisterBus({0x00: 0})
    configure_sensor(bus, 0x40)

    # INA3221 configuration 0x4127; SMBus word transport is byte-swapped.
    assert bus.writes == [(0x40, 0x00, 0x2741)]
