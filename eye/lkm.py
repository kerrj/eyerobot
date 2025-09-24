import serial
import time
from typing import Tuple


def _get_checksum(bytes):
    """
    Returns the checksum of a byte array
    """
    checksum = 0
    for i in bytes:
        checksum += i
    return checksum % 256


def _send_cmd(cmd: bytes, data: bytes, con: serial.Serial):
    """
    Sends the command payload, checks the checksum, reads the number of bytes from the response,
    and returns the raw data from the response (minus the checksum)
    """
    package = cmd + bytes([_get_checksum(cmd)])
    if len(data) > 0:
        package += data + bytes([_get_checksum(data)])
    con.write(package)
    resp = con.read(5)
    assert _get_checksum(resp[:-1]) == resp[-1]
    data_len = resp[3]
    if data_len > 0:
        response_data = con.read(data_len + 1)
        assert _get_checksum(response_data[:-1]) == response_data[-1]
        return response_data[:-1]


class LKMotor:
    def __init__(self, serial_con: serial.Serial, motor_id: int):
        self.serial_con = serial_con
        self.serial_con.set_low_latency_mode(True)
        self.motor_id = motor_id

    def _get_checksum(self, bytes_array):
        """
        Returns the checksum of a byte array
        """
        checksum = 0
        for i in bytes_array:
            checksum += i
        return checksum % 256

    def _send_command(self, cmd: int, data: bytes = b"") -> bytes:
        """
        Sends a command to the motor and returns the response data.

        :param cmd: Command byte
        :param data: Optional data bytes
        :return: Response data (excluding checksum)
        """
        # Construct the command frame
        cmd_frame = bytes([0x3E, cmd, self.motor_id, len(data)])

        # Use the _send_cmd helper function
        response = _send_cmd(cmd_frame, data, self.serial_con)

        return response

    # Basic motor control
    def turn_on(self) -> None:
        """Turn on the motor."""
        self._send_command(0x88)

    def turn_off(self) -> None:
        """Turn off the motor."""
        self._send_command(0x80)

    def stop(self) -> None:
        """Stop the motor."""
        self._send_command(0x81)

    # Position control
    def set_angle(self, angle: float, speed: float = 75000) -> None:
        """Set the motor angle (multi-loop)."""
        angle_value = int(angle * 100)  # Convert to 0.01 degree units
        speed = int(speed * 100)
        data = angle_value.to_bytes(8, "little", signed=True) + speed.to_bytes(
            4, "little"
        )
        self._send_command(0xA4, data)

    def set_single_loop_angle(
        self, angle: float, direction: int, speed: float = 75000
    ) -> None:
        """Set the motor angle (single-loop)."""
        angle_value = (
            int(angle * 100) % 36000
        )  # Convert to 0.01 degree units and ensure it's within 0-359.99
        speed = int(speed * 100)  # Convert to 0.01 dps units
        data = (
            bytes([direction])
            + angle_value.to_bytes(2, "little")
            + b"\x00"
            + speed.to_bytes(4, "little")
        )
        self._send_command(0xA6, data)

    def set_increment_angle(self, angle: float, speed: float = 75000) -> None:
        """Set the motor angle (incremental)."""
        angle_value = int(angle * 100)  # Convert to 0.01 degree units
        speed = int(speed * 100)
        data = angle_value.to_bytes(4, "little", signed=True) + speed.to_bytes(
            4, "little"
        )
        self._send_command(0xA8, data)

    # Speed control
    def set_speed(self, speed: float) -> None:
        """Set the motor speed."""
        speed_value = int(speed * 100)  # Convert to 0.01 dps units
        data = speed_value.to_bytes(4, "little", signed=True)
        self._send_command(0xA2, data)

    def clear_multi_loop(self):
        self._send_command(0x94)

    def set_open_loop_control(self, power_control: float) -> None:
        """
        Set the open loop control voltage for MS series motors.

        :param power_control: Power control value, range -1 to 1
        """
        if not -1.0 <= power_control <= 1.0:
            raise ValueError("Power control value must be between -1 and 1")

        # Convert power_control to bytes
        power_control = int(power_control * 850)
        power_bytes = power_control.to_bytes(2, "little", signed=True)

        # Send the command
        self._send_command(0xA0, power_bytes)

    # Reading motor state
    def get_angle(self) -> float:
        """Get the current motor angle."""
        response = self._send_command(0x92)
        angle_value = int.from_bytes(response[:8], "little", signed=True)
        return angle_value / 100  # Convert from 0.01 degree units to degrees

    def get_speed(self) -> float:
        """Get the current motor speed."""
        response = self._send_command(0x9C)
        speed_value = int.from_bytes(response[3:5], "little", signed=True)
        return speed_value  # already in dps

    def get_temperature(self) -> int:
        """Get the current motor temperature."""
        response = self._send_command(0x9A)
        return response[0]  # Temperature in Celsius

    def get_voltage(self) -> float:
        """Get the current motor voltage."""
        response = self._send_command(0x9A)
        voltage_value = int.from_bytes(response[1:3], "little")
        return voltage_value / 100  # Convert to volts

    # PID parameter adjustment
    def get_pid_parameters(self, param_id: int) -> Tuple[int, int, int]:
        """Get the PID parameters for a specific control loop."""
        response = self._send_command(0x40, bytes([0x00, param_id]))
        kp = int.from_bytes(response[1:3], "little")
        ki = int.from_bytes(response[3:5], "little")
        kd = int.from_bytes(response[5:7], "little")
        return kp, ki, kd

    def set_pid_parameters(
        self, param_id: int, kp: int, ki: int, kd: int, to_rom: bool = False
    ) -> None:
        """Set the PID parameters for a specific control loop."""
        data = (
            bytes([param_id])
            + kp.to_bytes(2, "little")
            + ki.to_bytes(2, "little")
            + kd.to_bytes(2, "little")
        )
        cmd = 0x44 if to_rom else 0x42
        self._send_command(cmd, data)


if __name__ == "__main__":
    import math

    con = serial.Serial("/dev/ttyUSB0", 2_000_000)
    print("Connected!")
    motor1 = LKMotor(con, 0x01)
    motor2 = LKMotor(con, 0x02)

    P_pos_1 = 500
    I_pos_1 = 10
    P_speed_1 = 500
    I_speed_1 = 10

    P_pos_2 = 500
    I_pos_2 = 10
    P_speed_2 = 500
    I_speed_2 = 10

    motor1.turn_on()
    motor2.turn_on()
    start = time.time()
    for i in range(1000):
        motor1.get_angle()
        motor2.get_angle()
    print("hz: ", 1000 / (time.time() - start))
    motor1.turn_off()
    motor2.turn_off()
