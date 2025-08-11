#!/usr/bin/env python3

"""
Hello World example for Boston Dynamics Spot SDK.

This script demonstrates the basic connection and interaction with a Spot robot.
It will connect to the robot, claim control, stand up, and then sit down.
"""

import argparse
import os
import sys
import time

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import robot_state_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

def hello_spot(config):
    """A simple Hello World program for Spot.
    
    This function will:
    1. Create a connection to the robot
    2. Authenticate
    3. Get robot state
    4. Acquire a lease
    5. Stand the robot up
    6. Wait a bit
    7. Sit the robot down
    8. Return the lease
    """
    # Create robot object and authenticate
    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot(config.hostname)
    
    try:
        bosdyn.client.util.authenticate(robot)
        robot.time_sync.wait_for_sync()
    except ResponseError as err:
        print(f"Failed to communicate with robot: {err}")
        return False

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), "Robot is estopped. Please use an external E-Stop client, " \
                                   "such as the estop SDK example, to configure E-Stop."

    # Get robot state client to check robot status
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    
    # Get current robot state
    robot_state = robot_state_client.get_robot_state()
    # print(f"{robot_state}")
    print(f"Battery: {robot_state.battery_states[0].charge_percentage.value}")
    print(f"Manipulator open percentage: {robot_state.manipulator_state.gripper_open_percentage}")
    

    # Only one client at a time can operate a robot. Clients acquire a lease to
    # indicate that they want to control a robot. Acquiring may fail if another
    # client is currently controlling the robot. When the client is done
    # controlling the robot, it should return the lease so other clients can
    # control it. The LeaseKeepAlive object takes care of keeping the lease alive
    # automatically.
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    
    try:
        with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            # Now, we are ready to power on the robot. This call will block until the power
            # is on. Commands would fail if this did not happen. We can also check that the robot is
            # powered at any point.
            print("Powering on robot... This may take several seconds.")
            robot.power_on(timeout_sec=20)
            assert robot.is_powered_on(), "Robot power on failed."
            print("Robot powered on.")

            # Tell the robot to stand up. The command service is used to issue commands to a robot.
            print("Commanding robot to stand...")
            command_client = robot.ensure_client(RobotCommandClient.default_service_name)
            blocking_stand(command_client, timeout_sec=10)
            print("Robot is standing.")

            # Wait a bit and then sit down
            print("Robot will sit down in 5 seconds...")
            time.sleep(5.0)
            
            # Now sit the robot down
            print("Commanding robot to sit...")
            sit_command = RobotCommandBuilder.synchro_sit_command()
            command_client.robot_command(sit_command)
            print("Robot is sitting.")

            # Power the robot off. By specifying "cut_immediately=False", a safe power off command
            # is issued to the robot. This will attempt to sit the robot before powering off.
            print("Powering off robot...")
            robot.power_off(cut_immediately=False, timeout_sec=20)
            assert not robot.is_powered_on(), "Robot power off failed."
            print("Robot safely powered off.")

    except Exception as exc:
        print(f"Hello, Spot! threw an exception: {exc}")
        return False

    return True


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hostname', help='hostname or IP address of robot', 
                       default=os.getenv('ROBOT_IP'))
    parser.add_argument('-u', '--username', help='username for robot authentication',
                       default=os.getenv('BOSDYN_CLIENT_USERNAME'))
    parser.add_argument('-p', '--password', help='password for robot authentication',
                       default=os.getenv('BOSDYN_CLIENT_PASSWORD'))
    parser.add_argument('-v', '--verbose', action='store_true', help='enable verbose output')
    options = parser.parse_args()
    
    # Validate required parameters
    if not all([options.hostname, options.username, options.password]):
        print("Error: hostname, username, and password are required")
        print("Provide them via command line or set environment variables:")
        print("  ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD")
        return 1

    try:
        if not hello_spot(options):
            return 1
        print("\nHello, Spot completed successfully!")
        return 0
    except Exception as exc:
        print(f"Hello, Spot! threw an exception: {exc}")
        return 1


if __name__ == '__main__':
    if not main():
        sys.exit(1)