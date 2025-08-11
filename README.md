# Spot SDK Project

This repository uses the Boston Dynamics Spot SDK as a git subrepo to develop applications for the Spot robot.

## Prerequisites

- Python 3.8 or higher
- Git
- Access to a Boston Dynamics Spot robot (physical or simulator)
- Valid Spot SDK license and credentials

## Setup Instructions

### 1. Clone this repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Initialize and update submodules

Initialize the Spot SDK submodule:

```bash
# Initialize and update submodules (if cloning an existing repo with submodules)
git submodule update --init --recursive
```

### 3. Install Python dependencies

Create a virtual environment and install the required packages:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install additional dependencies for this project
python3 -m pip install -r requirements.txt  # if you have project-specific requirements
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```bash
# Robot connection details
BOSDYN_CLIENT_USERNAME=your_username
BOSDYN_CLIENT_PASSWORD=your_password
ROBOT_IP=192.168.80.3  # Replace with your robot's IP
```

**Important**: Never expose your `.env` file. Add it to your `.gitignore` to keep credentials secure.

## Running the Hello World Example

The hello world example demonstrates basic connection and interaction with the Spot robot:

```bash
python hello_odense.py --hostname ROBOT_IP --username USERNAME --password PASSWORD
```

Or if you've set up the environment variables:

```bash
python hello_odense.py
```

## Running a Spot SDK Example

To run an example from the Spot SDK, navigate to the `spot-sdk/python/examples` directory and execute the desired script. For example:

```bash
cd spot-sdk/python/examples/hello_spot
# Install example dependencies if needed
python3 -m pip install -r requirements.txt
python hello_spot.py <ROBOT_IP>
```

## Project Structure

```
.
├── README.md                 # This file
├── hello_odense.py            # Hello world example
├── requirements.txt         # Project dependencies
├── .env                     # Environment variables (not in git)
├── .gitignore              # Git ignore file
├── .gitmodules             # Git submodules configuration
└── spot-sdk/               # Spot SDK submodule
    ├── python/             # Python SDK
    ├── docs/               # Documentation
    └── ...                 # Other SDK files
```

## Working with Submodules

### Cloning a repository with submodules

When someone else clones your repository, they need to initialize the submodules:

```bash
git clone --recursive <your-repository-url>
# OR if already cloned:
git submodule update --init --recursive
```

### Updating the Spot SDK submodule

To update to a newer version of the Spot SDK:

```bash
cd spot-sdk
git fetch
git checkout v4.1.1  # or newer version tag
cd ..
git add spot-sdk
git commit -m "Update Spot SDK to v4.1.1"
```

### Working with submodule changes

If you make changes within the submodule:

```bash
cd spot-sdk
# Make your changes
git add .
git commit -m "Your changes"
git push origin HEAD:your-branch-name
cd ..
git add spot-sdk
git commit -m "Update submodule reference"
```

## Development Tips

1. **Robot Safety**: Always ensure the robot has clear space around it before running programs
2. **Emergency Stop**: Keep the emergency stop button accessible at all times
3. **Testing**: Test your code in simulation first when possible
4. **Logging**: Enable appropriate logging levels for debugging

## Common Issues

### Connection Problems
- Verify robot IP address and network connectivity
- Check username/password credentials
- Ensure robot is powered on and ready
- Verify firewall settings allow connection on required ports

### SDK Installation Issues
- Make sure you're using Python 3.8+
- Try installing in a clean virtual environment
- Check that all requirements.txt dependencies are satisfied

## Resources

- [Official Spot SDK Documentation](https://dev.bostondynamics.com/)
- [Spot SDK Python Examples](https://github.com/boston-dynamics/spot-sdk/tree/master/python/examples)
- [Boston Dynamics Developer Portal](https://dev.bostondynamics.com/)
- [Spot SDK GitHub Issues](https://github.com/boston-dynamics/spot-sdk/issues)

## License

This project follows the same license terms as the Boston Dynamics Spot SDK. See the `spot-sdk/LICENSE` file for details.

## Contributing

1. Fork this repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly with a robot
5. Submit a pull request

## Safety Notice

Always follow Boston Dynamics safety guidelines when operating Spot. The robot is powerful and can cause injury if not operated properly. Never run untested code on the robot without proper safety precautions.