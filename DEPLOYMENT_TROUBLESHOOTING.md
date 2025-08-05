# Deployment Troubleshooting Guide

## Requirements Installation Error Solutions

### 1. For Streamlit Cloud Deployment:
Use the simplified requirements file:
```
cp requirements-simple.txt requirements.txt
```

### 2. For Heroku Deployment:
Ensure you have both files:
- `requirements.txt` (with package versions)
- `runtime.txt` (with Python version)

### 3. For Local Docker Deployment:
Use the standard requirements.txt with specific versions.

### 4. Common Issues and Solutions:

#### Issue: Package Version Conflicts
**Solution A**: Use simple requirements without version pinning
```
streamlit
pandas
numpy
scikit-learn
plotly
matplotlib
seaborn
```

**Solution B**: Use compatible version ranges
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.21.0
scikit-learn>=1.3.0
plotly>=5.15.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

#### Issue: Python Version Compatibility
**Solution**: Create runtime.txt with supported Python version
```
python-3.12.3
```
Or try more compatible versions:
```
python-3.11.5
python-3.10.12
```

#### Issue: Memory/Build Timeout
**Solution**: Use minimal requirements and reduce package sizes

### 5. Platform-Specific Requirements:

#### Streamlit Cloud:
- Use requirements.txt (no versions needed)
- Python 3.9-3.12 supported
- Main app file should be in root directory

#### Heroku:
- Requires Procfile
- Needs runtime.txt
- Specific buildpack may be needed

#### Render:
- Similar to Heroku
- Auto-detects Python apps
- Uses requirements.txt

### 6. Debug Steps:
1. Check deployment logs for specific error messages
2. Try with simplified requirements first
3. Verify Python version compatibility
4. Ensure main app file is accessible
5. Check file paths are correct

### 7. Alternative Requirements Files:

We've created multiple options:
- `requirements.txt` - Main file with version ranges
- `requirements-simple.txt` - No versions (most compatible)
- You can rename the simple one if needed:
```bash
mv requirements-simple.txt requirements.txt
```

### 8. Test Locally First:
```bash
pip install -r requirements.txt
streamlit run APP.phase1.streamlit.py
```

If it works locally but fails on deployment, it's usually a platform-specific issue.
