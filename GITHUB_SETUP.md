# GitHub Setup Guide

Follow these steps to upload your project to GitHub:

## Step 1: Create a GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click the **"+"** button in the top right corner
3. Select **"New repository"**
4. Fill in the repository details:
   - **Repository name**: `dio-optimization` (or your preferred name)
   - **Description**: "Dholes-Inspired Optimization for Feature Selection and Hyperparameter Tuning"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click **"Create repository"**

## Step 2: Initialize Git (If Not Already Done)

Open PowerShell in your project directory and run:

```powershell
cd C:\Users\LENOVO\Desktop\Dio_expose
git init
```

## Step 3: Configure Git (First Time Only)

If you haven't configured git before, set your name and email:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 4: Add Files to Git

```powershell
# Add all files
git add .

# Or add specific files
git add dio.py main.py README.md requirements.txt .gitignore LICENSE
```

## Step 5: Create First Commit

```powershell
git commit -m "Initial commit: DIO optimization for feature selection and hyperparameter tuning"
```

## Step 6: Connect to GitHub Repository

Replace `YOUR_USERNAME` with your actual GitHub username:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/dio-optimization.git
```

## Step 7: Push to GitHub

```powershell
# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 8: Verify Upload

1. Go to your GitHub repository URL
2. You should see all your files uploaded
3. The README.md will be displayed on the main page

---

## Troubleshooting

### Authentication Required

If GitHub asks for authentication:

**Option 1: Personal Access Token (Recommended)**
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` scope
3. Use the token as your password when pushing

**Option 2: GitHub CLI**
```powershell
# Install GitHub CLI from https://cli.github.com/
gh auth login
```

### If Repository Already Exists

If you need to force push (use with caution):
```powershell
git push -f origin main
```

---

## Future Updates

To push changes after the initial upload:

```powershell
# Add changed files
git add .

# Commit changes
git commit -m "Description of changes"

# Push to GitHub
git push
```

---

## Optional: Add .gitattributes for Large Files

If you have large output files, create a `.gitattributes` file:

```
*.png filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
*.pdf filter=lfs diff=lfs merge=lfs -text
```

Then install Git LFS:
```powershell
git lfs install
```

---

## Quick Reference Commands

```powershell
# Check status
git status

# View commit history
git log

# View remote repository
git remote -v

# Pull latest changes
git pull

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
```
