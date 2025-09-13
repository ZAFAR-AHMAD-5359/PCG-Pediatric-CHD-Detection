# Contributing to Cardiac Signal Processing Projects

Thank you for your interest in contributing to our cardiac signal processing research! This document provides guidelines for contributing to our repositories.

## 🤝 Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## 🚀 How to Contribute

### 1. Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

**Bug Report Template:**
```markdown
**Description:** Clear description of the bug
**Steps to Reproduce:**
1. Step one
2. Step two
3. ...

**Expected Behavior:** What should happen
**Actual Behavior:** What actually happens
**Environment:**
- OS: [e.g., Ubuntu 20.04]
- Python Version: [e.g., 3.8.5]
- TensorFlow Version: [e.g., 2.8.0]
```

### 2. Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues.

**Enhancement Template:**
```markdown
**Use Case:** Why is this enhancement needed?
**Proposed Solution:** Your suggested implementation
**Alternatives:** Other approaches considered
**Additional Context:** Any other relevant information
```

### 3. Pull Requests

1. **Fork the Repository**
   ```bash
   git clone https://github.com/ZAFAR-AHMAD-5359/[repo-name].git
   cd [repo-name]
   git remote add upstream https://github.com/ZAFAR-AHMAD-5359/[repo-name].git
   ```

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

   Commit message format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `style:` Code style changes
   - `refactor:` Code refactoring
   - `test:` Test additions/changes
   - `chore:` Maintenance tasks

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Coding Standards

### Python Style Guide

We follow PEP 8 with the following specifications:

```python
# Good example
def process_pcg_signal(signal: np.ndarray,
                       sample_rate: int = 4000,
                       filter_range: tuple = (20, 600)) -> np.ndarray:
    """
    Process PCG signal with bandpass filtering.

    Args:
        signal: Input PCG signal
        sample_rate: Sampling frequency in Hz
        filter_range: (low, high) cutoff frequencies

    Returns:
        Filtered signal array
    """
    # Implementation here
    pass
```

### Documentation Requirements

All functions must include:
- Brief description
- Args with types
- Returns description
- Example usage (for complex functions)

### Testing Guidelines

```python
# Test file structure
test_module_name.py

# Test function naming
def test_function_name_expected_behavior():
    # Arrange
    input_data = prepare_test_data()

    # Act
    result = function_under_test(input_data)

    # Assert
    assert result == expected_output
```

## 🔬 Research Contributions

For contributions involving new algorithms or research:

1. **Provide Citations**: Include relevant papers
2. **Benchmark Results**: Compare with existing methods
3. **Reproducibility**: Ensure results can be reproduced
4. **Documentation**: Explain the theoretical background

## 📊 Data Contributions

If contributing datasets:

1. **Ethical Approval**: Ensure proper ethics clearance
2. **Anonymization**: Remove all patient identifiers
3. **Documentation**: Include data collection protocol
4. **Format**: Use standard formats (WAV for audio, CSV for annotations)

## 🏥 Clinical Validation

For clinically relevant contributions:

1. **Medical Accuracy**: Verify with medical literature
2. **Safety Considerations**: Note any limitations
3. **Disclaimer**: Include appropriate medical disclaimers

## 📋 Checklist Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] PR description explains changes
- [ ] No sensitive data included
- [ ] Dependencies are documented

## 🎯 Priority Areas

We especially welcome contributions in:

1. **Performance Optimization**: Speed improvements
2. **New Cardiac Conditions**: Additional disease detection
3. **Noise Robustness**: Better noise handling
4. **Mobile Deployment**: Edge device optimization
5. **Clinical Validation**: Real-world testing

## 📚 Resources

- [Signal Processing Basics](https://www.scipy.org/docs.html)
- [Deep Learning for Healthcare](https://www.nature.com/articles/s41591-018-0316-z)
- [PCG Analysis Review](https://www.sciencedirect.com/science/article/pii/S0010482518300337)

## 📧 Questions?

Feel free to:
- Open an issue for discussions
- Email: zafarahmad5359@gmail.com
- Connect on [LinkedIn](https://linkedin.com)

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Acknowledged in research papers (for significant contributions)
- Included in release notes

Thank you for helping advance cardiac healthcare through technology! 💓