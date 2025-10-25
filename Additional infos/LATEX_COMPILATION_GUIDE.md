# LaTeX Report Compilation Guide

## 📄 Overview

The file `report.tex` is now a comprehensive LaTeX document containing all details of your DIO research project. This guide will help you compile it into a professional PDF.

---

## 🚀 Quick Start

### Option 1: Overleaf (Recommended - Easiest)

1. **Go to**: [https://www.overleaf.com](https://www.overleaf.com)
2. **Sign up/Login** (free account works fine)
3. **Create New Project** → Upload Project
4. **Upload** your `report.tex` file
5. **Also upload**:
   - `statistical_comparison_visualization.png`
   - `individual_model_trends.png`
   - `benchmark_visualization.png`
   - Any Visio diagrams you create (as PNG or PDF)
6. **Click "Recompile"** → PDF appears on the right!

**Advantages**: No installation needed, works in browser, automatic compilation, collaboration features

---

### Option 2: Local Installation

#### Windows

1. **Install MiKTeX**:
   - Download from: [https://miktex.org/download](https://miktex.org/download)
   - Run installer (accepts defaults)
   - MiKTeX will auto-install missing packages

2. **Install TeXstudio** (optional but recommended):
   - Download from: [https://www.texstudio.org](https://www.texstudio.org)
   - Provides a nice editor with built-in PDF viewer

3. **Compile**:
   - Open `report.tex` in TeXstudio
   - Press F5 (or click green arrow)
   - PDF appears automatically

**Command Line Alternative**:
```powershell
cd C:\Users\LENOVO\Desktop\Dio_expose
pdflatex report.tex
pdflatex report.tex  # Run twice for references
```

#### Linux

```bash
# Install TeX Live
sudo apt-get install texlive-full  # Ubuntu/Debian
# or
sudo dnf install texlive-scheme-full  # Fedora

# Compile
cd /path/to/Dio_expose
pdflatex report.tex
pdflatex report.tex  # Run twice
```

#### macOS

```bash
# Install MacTeX
brew install --cask mactex

# Compile
cd /path/to/Dio_expose
pdflatex report.tex
pdflatex report.tex
```

---

## 📋 Checklist Before Compiling

### ✅ Files Needed

Make sure these files are in the same directory as `report.tex`:

- [ ] `statistical_comparison_visualization.png` ✅ (already generated)
- [ ] `individual_model_trends.png` ✅ (already generated)
- [ ] `benchmark_visualization.png` ✅ (already generated)
- [ ] Visio diagrams (create these - see below)

### ✅ TODO Items in report.tex

The report has placeholders for content you need to add:

1. **Author Names** (Line ~87-90):
   ```latex
   \author[1]{Your Name}  % ← Replace with your actual name
   \author[2]{Professor's Name}  % ← Replace
   \affil[1]{Your Department/Affiliation}
   \affil[2]{Professor's Department/Affiliation}
   ```

2. **Code Snippet** (Line ~178):
   ```latex
   \textbf{TODO: Insert Code Snippet Here}
   ```
   - Copy a relevant section from `dio.py` or `main.py`
   - Use the `\begin{lstlisting}...\end{lstlisting}` environment

3. **Visio Diagrams** (Multiple locations):
   ```latex
   \framebox(300,200){Placeholder for Visio Diagram}
   ```
   - Create diagrams following `VISIO_SCHEMA_GUIDE.md`
   - Export as PNG or PDF
   - Replace `\framebox` with `\includegraphics`

---

## 🎨 Adding Your Content

### How to Add Code Snippets

Find this section in `report.tex` (around line 178):

```latex
% --- Placeholder for Code Snippet ---
\begin{figure}[H]
    \centering
    \rule{12cm}{0.1pt}
    \caption*{
        \textbf{TODO: Insert Code Snippet Here} \\
        ...
    }
    \rule{12cm}{0.1pt}
\end{figure}
```

**Replace with**:

```latex
\begin{lstlisting}[language=Python, caption={DIO Position Update Method}]
def _update_position(self, dhole_idx, alpha_pos, population):
    current_pos = population[dhole_idx]
    
    # Strategy 1: Chase alpha
    r1 = np.random.rand(len(current_pos))
    X_chase = alpha_pos + r1 * (alpha_pos - current_pos)
    
    # Strategy 2: Random pack member
    r2 = np.random.rand(len(current_pos))
    random_idx = np.random.randint(0, len(population))
    X_random = population[random_idx] + r2 * (population[random_idx] - current_pos)
    
    # Strategy 3: Pack center
    r3 = np.random.rand(len(current_pos))
    X_mean = np.mean(population, axis=0)
    X_scavenge = X_mean + r3 * (X_mean - current_pos)
    
    # Average of three strategies
    new_pos = (X_chase + X_random + X_scavenge) / 3
    
    return np.clip(new_pos, self.lower_bound, self.upper_bound)
\end{lstlisting}
```

### How to Add Visio Diagrams

1. **Create diagrams** in Visio (see `VISIO_SCHEMA_GUIDE.md`)
2. **Export** each as PNG or PDF (300 DPI recommended)
3. **Save** in the same folder as `report.tex`
4. **Find placeholders** in `report.tex`:

```latex
\begin{figure}[H]
    \centering
    \framebox(300,200){Placeholder for Visio Diagram}
    \caption{...}
    \label{fig:nested_loop}
\end{figure}
```

5. **Replace with**:

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.9\textwidth]{nested_optimization_diagram.png}
    \caption{Nested optimization structure showing hierarchical DIO loops for hyperparameter tuning and feature selection.}
    \label{fig:nested_loop}
\end{figure}
```

---

## 🔧 Common Compilation Issues

### Issue 1: Missing Packages

**Error**: `! LaTeX Error: File 'booktabs.sty' not found.`

**Solution**:
- **MiKTeX**: Should auto-install. Check MiKTeX Console → Packages
- **Manual**: Install texlive-latex-extra
- **Overleaf**: Should work automatically

### Issue 2: Images Not Found

**Error**: `! Package pdftex.def Error: File 'statistical_comparison_visualization.png' not found.`

**Solution**:
- Ensure PNG files are in the same directory as `report.tex`
- Check file names match exactly (case-sensitive on Linux/Mac)
- Use forward slashes in paths: `images/diagram.png`

### Issue 3: References Not Showing

**Solution**: Run pdflatex **twice**
```bash
pdflatex report.tex  # First pass
pdflatex report.tex  # Second pass - resolves references
```

### Issue 4: Table of Contents Not Updating

**Solution**: Delete auxiliary files and recompile
```bash
rm report.aux report.toc  # Linux/Mac
del report.aux report.toc  # Windows
pdflatex report.tex
pdflatex report.tex
```

---

## 📊 Expected Output

After successful compilation, you should see:

- **Title Page**: With your name and professor's name
- **Abstract**: One-page summary
- **Table of Contents**: Automatic, with page numbers
- **Section 1 - Introduction**: 2-3 pages
- **Section 2 - Background**: 4-5 pages with equations
- **Section 3 - Framework**: 3-4 pages with diagrams
- **Section 4 - Results**: 5-6 pages with tables and figures
- **Section 5 - Conclusion**: 2-3 pages
- **Acknowledgments**: 1 paragraph
- **References**: 10 citations
- **Appendices**: Code, tables, hyperparameter details

**Total**: Approximately 20-25 pages

---

## 🎯 Final Steps

### Before Submission

1. **Proofread** all content
2. **Check figures**:
   - All images display correctly
   - Captions are descriptive
   - Figure numbers referenced in text
3. **Verify tables**:
   - Data matches your CSV files
   - Formatting is clean
4. **Check references**:
   - All citations present
   - DOIs/URLs work
5. **Review equations**:
   - Rendered correctly
   - Variables defined
6. **Test PDF**:
   - All links work (table of contents, references)
   - No blank pages
   - Page numbers correct

### Export Options

From your PDF viewer or LaTeX editor:

- **Print to PDF**: For final submission version
- **Export with bookmarks**: For easier navigation
- **Compress images**: If file size is large (>10MB)

---

## 💡 Tips for Professional Output

1. **Consistent Formatting**:
   - All figure captions below figures
   - All table captions above tables
   - Consistent font sizes

2. **High-Quality Images**:
   - Export Visio diagrams at 300 DPI
   - Use vector formats (PDF) when possible
   - Resize images proportionally

3. **Mathematical Notation**:
   - Use `\texttt{}` for code/variables
   - Use `$...$` for inline math
   - Use `\begin{equation}...\end{equation}` for display math

4. **Cross-References**:
   - Use `\ref{label}` to reference figures/tables
   - Example: "as shown in Figure \ref{fig:pareto}"

---

## 📚 Additional Resources

- **LaTeX Tutorial**: [https://www.overleaf.com/learn/latex/Tutorials](https://www.overleaf.com/learn/latex/Tutorials)
- **LaTeX Wikibook**: [https://en.wikibooks.org/wiki/LaTeX](https://en.wikibooks.org/wiki/LaTeX)
- **Math Symbols**: [http://detexify.kirelabs.org/classify.html](http://detexify.kirelabs.org/classify.html) (draw symbol to find code)
- **Tables Generator**: [https://www.tablesgenerator.com/](https://www.tablesgenerator.com/)

---

## 🆘 Getting Help

If you encounter issues:

1. **Check error message**: LaTeX errors usually show line numbers
2. **Google the error**: Most common errors have Stack Exchange answers
3. **Overleaf help**: [https://www.overleaf.com/learn](https://www.overleaf.com/learn)
4. **TeX Stack Exchange**: [https://tex.stackexchange.com/](https://tex.stackexchange.com/)

---

## ✅ Summary

Your `report.tex` is now a **complete, publication-ready document** with:

- ✅ Comprehensive introduction and background
- ✅ Detailed methodology from the original DIO paper
- ✅ Random Forest architecture explanation
- ✅ Nested optimization framework
- ✅ Complete experimental setup
- ✅ Statistical results and analysis
- ✅ Limitations and future work
- ✅ Expanded conclusion
- ✅ 10 references
- ✅ 3 appendices with code and data

**Just add**:
1. Your name and affiliation
2. Code snippet (optional)
3. Visio diagrams (7 total)

Then compile and you're ready to submit! 🎓📝🏆

---

**Last Updated**: October 25, 2025  
**Status**: Ready for compilation and submission
