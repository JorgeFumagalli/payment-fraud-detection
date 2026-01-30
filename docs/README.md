# Documentation

This directory contains comprehensive documentation for the Payment Fraud Detection System.

## Available Documents

### 📘 Technical Documentation

- **[deployment_guide.md](deployment_guide.md)** - Complete guide for deploying to production
  - Environment setup
  - Deployment options (batch, API, Docker)
  - Monitoring and maintenance
  - Troubleshooting

### 📊 Case Study (Upload Required)

- **case_study.pdf** - 32-page technical analysis
  - Project methodology
  - Feature engineering details
  - Model comparison
  - Business impact analysis
  - ROI projections
  
  *Note: Please add your case study PDF to this directory.*

### 📚 Additional Resources

Coming soon:
- **api_reference.md** - Complete API documentation
- **model_comparison.md** - Detailed model performance analysis
- **features.md** - Feature catalog with descriptions
- **tutorials/** - Step-by-step tutorials and notebooks

## Quick Links

- [Main README](../README.md) - Project overview
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Changelog](../CHANGELOG.md) - Version history

## Building Documentation

If using Sphinx for documentation:

```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Build HTML documentation
cd docs
sphinx-build -b html . _build

# View documentation
open _build/index.html
```

## Documentation Standards

When contributing documentation:

1. **Use Markdown** for all docs (except API reference)
2. **Follow structure**: Introduction → Prerequisites → Steps → Examples
3. **Include code examples** with syntax highlighting
4. **Add diagrams** where helpful (use Mermaid or PlantUML)
5. **Keep it updated** - sync with code changes

## Need Help?

- Open an issue on GitHub
- Email: jfumagalli.work@gmail.com
- Check existing documentation first

---

*Last updated: January 2026*
