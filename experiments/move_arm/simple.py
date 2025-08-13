import argparse
import sys
import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.lease import ResourceAlreadyClaimedError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand


def stand_then_sit(config):
    bosdyn.client.util.setup_logging(config.verbose)
    sdk = bosdyn.client.create_standard_sdk('StandThenSitClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client.'

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    try:
        # Try to acquire lease normally
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            _run_robot_commands(robot)
    except ResourceAlreadyClaimedError:
        robot.logger.warning("Lease already claimed. Attempting to forcefully take lease...")

        # Forcefully take lease
        lease = lease_client.take_lease()

        # Now maintain the taken lease with LeaseKeepAlive, passing the lease explicitly
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, lease=lease, must_acquire=False, return_at_exit=True):
            _run_robot_commands(robot)


def _run_robot_commands(robot):
    robot.logger.info('Powering on robot...')
    robot.power_on(timeout_sec=20)
    assert robot.is_powered_on(), 'Robot power on failed.'
    robot.logger.info('Robot powered on.')

    robot.logger.info('Commanding robot to stand...')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    blocking_stand(command_client, timeout_sec=10)
    robot.logger.info('Robot standing.')

    time.sleep(60)  # Stand for 60 seconds

    robot.logger.info('Commanding robot to sit...')
    sit_command = RobotCommandBuilder.synchro_sit_command()
    command_client.robot_command(sit_command)
    time.sleep(5)  # Wait for robot to sit down

    robot.logger.info('Powering off robot...')
    robot.power_off(cut_immediately=False, timeout_sec=20)
    assert not robot.is_powered_on(), 'Robot power off failed.'
    robot.logger.info('Robot safely powered off.')


def main():
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args()
    try:
        stand_then_sit(options)
        return True
    except Exception as exc:
        logger = bosdyn.client.util.get_logger()
        logger.error('StandThenSit threw an exception: %r', exc)
        return False


if __name__ == '__main__':
    if not main():
        sys.exit(1)
