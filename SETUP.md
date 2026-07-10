# Setup Guide

This guide will help you set up the Multi-Agent Collusion & Correlated Failure Benchmark on your machine.

## Prerequisites

- Python 3.10 or higher
- Git
- pip (Python package manager)

## Mac Setup

### Step 1: Install Python

If you don't have Python 3.10+ installed:

```bash
# Using Homebrew (recommended)
brew install python@3.12

# Or download from https://www.python.org/downloads/
```

Verify installation:
```bash
python3 --version
# Should show Python 3.10 or higher
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/yourusername/multicollude.git
cd multicollude
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Install the package in development mode
pip install -e ".[dev]"
```

### Step 5: Verify Installation

```bash
# Run the test suite
pytest tests/

# You should see all 37 tests passing
```

### Step 6: Run Your First Experiment

```bash
# Run a simple market scenario
multicollude run --config configs/scenarios/market_bertrand_2f.yaml
```

## Windows Setup

### Step 1: Install Python

1. Download Python 3.12 from https://www.python.org/downloads/
2. Run the installer
3. **Important:** Check "Add Python to PATH" during installation

Verify installation in Command Prompt or PowerShell:
```cmd
python --version
:: Should show Python 3.10 or higher
```

### Step 2: Clone the Repository

```cmd
git clone https://github.com/yourusername/multicollude.git
cd multicollude
```

### Step 3: Create Virtual Environment

```cmd
:: Create virtual environment
python -m venv venv

:: Activate it (Command Prompt)
venv\Scripts\activate

:: Or activate it (PowerShell)
venv\Scripts\Activate.ps1
```

**Note:** If you get a permission error in PowerShell, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Step 4: Install Dependencies

```cmd
:: Install the package in development mode
pip install -e ".[dev]"
```

### Step 5: Verify Installation

```cmd
:: Run the test suite
pytest tests/

:: You should see all 37 tests passing
```

### Step 6: Run Your First Experiment

```cmd
:: Run a simple market scenario
multicollude run --config configs/scenarios/market_bertrand_2f.yaml
```

## Project Structure

After setup, your directory should look like this:

```
multicollude/
├── venv/                    # Virtual environment (gitignored)
├── src/multicollude/        # Main package
│   ├── core/                # Environment engine
│   ├── agents/              # LLM agent implementations
│   ├── scenarios/           # Environment implementations
│   ├── metrics/             # Metrics engine
│   └── detection/           # Detection track
├── configs/                 # Scenario configurations
├── tests/                   # Test suite
├── pyproject.toml           # Project configuration
├── README.md                # Main documentation
├── PLAN.md                  # Project plan
└── SETUP.md                 # This file
```

## Common Issues

### Issue: "pip install" fails with permission error

**Solution:** Use the `--user` flag or activate your virtual environment first:
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Then install
pip install -e ".[dev]"
```

### Issue: Python version too old

**Solution:** Install Python 3.10 or higher:
- Mac: `brew install python@3.12`
- Windows: Download from https://www.python.org/downloads/

### Issue: Tests fail with import errors

**Solution:** Make sure the virtual environment is activated and the package is installed:
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -e ".[dev]"
pytest tests/
```

### Issue: "multicollude" command not found

**Solution:** Make sure you're in the virtual environment:
```bash
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# The command should now work
multicollude --help
```

## Next Steps

Once setup is complete:

1. **Read the plan:** See `PLAN.md` for the project overview
2. **Explore the code:** Start with `src/multicollude/core/engine.py`
3. **Run experiments:** Try different configurations in `configs/`
4. **Run tests:** `pytest tests/ -v` to see all tests

## Getting Help

- Check the [README.md](README.md) for quick start
- Read [PLAN.md](PLAN.md) for the project plan
- Open an issue on GitHub
