# Contributing to Payment Fraud Detection

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)

## 📜 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive criticism
- Respect differing viewpoints and experiences

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/payment-fraud-detection.git
   cd payment-fraud-detection
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/originaluser/payment-fraud-detection.git
   ```

## 💻 Development Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   pip install -e .  # Install package in editable mode
   ```

3. **Verify installation**:
   ```bash
   python -c "import src; print('✅ Installation successful!')"
   ```

## 🔨 Making Changes

### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Workflow

1. **Create a new branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```

2. **Make your changes** following the [code style guidelines](#code-style)

3. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add: Brief description of changes"
   ```

   **Commit message format**:
   - `Add:` for new features
   - `Fix:` for bug fixes
   - `Update:` for updates to existing features
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring

4. **Keep your branch updated**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_features.py

# Run with verbose output
pytest tests/ -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names: `test_feature_engineer_creates_correct_features()`
- Include docstrings explaining what the test validates

Example:
```python
def test_feature_engineer_creates_25_features():
    """Test that FeatureEngineer creates exactly 25 features."""
    engineer = FeatureEngineer()
    features = engineer.get_feature_names()
    assert len(features) == 25, f"Expected 25 features, got {len(features)}"
```

## 🎨 Code Style

### Python Style Guide

We follow **PEP 8** with these tools:

1. **Black** (code formatter):
   ```bash
   black src/
   ```

2. **Flake8** (linter):
   ```bash
   flake8 src/ --max-line-length=127
   ```

3. **MyPy** (type checker):
   ```bash
   mypy src/ --ignore-missing-imports
   ```

### Documentation Style

- Use **NumPy-style docstrings**:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Brief one-line description.
    
    More detailed explanation if needed.
    
    Parameters
    ----------
    param1 : int
        Description of param1
    param2 : str
        Description of param2
        
    Returns
    -------
    bool
        Description of return value
        
    Example
    -------
    >>> result = example_function(42, "test")
    >>> print(result)
    True
    """
    return True
```

### Code Quality Checklist

Before submitting:
- [ ] Code is formatted with Black
- [ ] No Flake8 warnings
- [ ] All tests pass
- [ ] New code has tests
- [ ] Docstrings are complete
- [ ] Type hints are included
- [ ] Comments explain "why", not "what"

## 📤 Submitting Changes

### Pull Request Process

1. **Push your branch**:
   ```bash
   git push origin feature/amazing-feature
   ```

2. **Create Pull Request** on GitHub:
   - Provide clear title and description
   - Reference related issues (e.g., "Fixes #123")
   - Explain what changed and why
   - Include screenshots for UI changes

3. **PR Description Template**:
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Code refactoring
   
   ## Testing
   Describe how you tested these changes
   
   ## Checklist
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation updated
   - [ ] No new warnings
   ```

4. **Address review feedback**:
   - Make requested changes
   - Push additional commits to the same branch
   - Respond to reviewer comments

## 🎯 Areas for Contribution

We welcome contributions in these areas:

### High Priority
- [ ] Additional model architectures (CatBoost, LightGBM)
- [ ] Real-time streaming prediction API
- [ ] Dashboard for model monitoring
- [ ] A/B testing framework

### Medium Priority
- [ ] Additional feature engineering techniques
- [ ] Automated model retraining pipeline
- [ ] Model explainability enhancements
- [ ] Performance optimization

### Documentation
- [ ] Tutorial notebooks
- [ ] API documentation improvements
- [ ] Use case examples
- [ ] Video walkthroughs

### Testing
- [ ] Increase test coverage
- [ ] Integration tests
- [ ] Performance benchmarks
- [ ] Edge case testing

## 📧 Questions?

- Open an issue for bugs or feature requests
- Use discussions for general questions
- Tag maintainers for urgent matters

## 🙏 Thank You!

Your contributions make this project better for everyone. We appreciate your time and effort!

---

*This guide is inspired by open-source best practices and is subject to updates.*
