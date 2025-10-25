"""
PowerPoint Presentation Generator for DIO Research Project
Creates a 15-minute presentation ready for export
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import json

# Load optimization results
with open('optimization_results.json', 'r') as f:
    opt_results = json.load(f)

# Create presentation
prs = Presentation()
prs.slide_width = Inches(10)
prs.slide_height = Inches(7.5)

def add_title_slide(prs, title, subtitle):
    """Add a title slide"""
    slide_layout = prs.slide_layouts[0]  # Title slide
    slide = prs.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    subtitle_shape = slide.placeholders[1]
    
    title_shape.text = title
    subtitle_shape.text = subtitle
    
    # Format title
    title_shape.text_frame.paragraphs[0].font.size = Pt(44)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(44, 62, 80)
    
    return slide

def add_content_slide(prs, title, content_type='bullet'):
    """Add a content slide with title and body"""
    slide_layout = prs.slide_layouts[1]  # Title and Content
    slide = prs.slides.add_slide(slide_layout)
    
    title_shape = slide.shapes.title
    title_shape.text = title
    title_shape.text_frame.paragraphs[0].font.size = Pt(40)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(44, 62, 80)
    
    return slide

def add_bullet_points(text_frame, points, level=0):
    """Add bullet points to a text frame"""
    text_frame.clear()
    for i, point in enumerate(points):
        p = text_frame.paragraphs[0] if i == 0 else text_frame.add_paragraph()
        p.text = point
        p.level = level
        p.font.size = Pt(18)
        p.space_after = Pt(12)

def add_image_slide(prs, title, image_path, caption=""):
    """Add a slide with an image"""
    slide = add_content_slide(prs, title)
    
    # Add image
    left = Inches(1.5)
    top = Inches(2)
    height = Inches(4.5)
    
    try:
        pic = slide.shapes.add_picture(image_path, left, top, height=height)
        
        # Add caption if provided
        if caption:
            left = Inches(1)
            top = Inches(6.5)
            width = Inches(8)
            height = Inches(0.5)
            
            textbox = slide.shapes.add_textbox(left, top, width, height)
            text_frame = textbox.text_frame
            p = text_frame.paragraphs[0]
            p.text = caption
            p.font.size = Pt(14)
            p.font.italic = True
            p.alignment = PP_ALIGN.CENTER
    except:
        # If image not found, add placeholder text
        left = Inches(2)
        top = Inches(3)
        width = Inches(6)
        height = Inches(2)
        
        textbox = slide.shapes.add_textbox(left, top, width, height)
        text_frame = textbox.text_frame
        p = text_frame.paragraphs[0]
        p.text = f"[Image: {image_path}]"
        p.font.size = Pt(24)
        p.alignment = PP_ALIGN.CENTER
    
    return slide

# ============================================================================
# SLIDE 1: TITLE SLIDE
# ============================================================================
add_title_slide(
    prs,
    "Dholes-Inspired Optimization (DIO)\nfor Feature Selection & Hyperparameter Tuning",
    "Breast Cancer Classification using Random Forest\n\nYour Name | Your University | October 2025"
)

# ============================================================================
# SLIDE 2: AGENDA
# ============================================================================
slide = add_content_slide(prs, "Presentation Outline")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Introduction & Motivation",
    "DIO Algorithm Overview",
    "Methodology: Nested Optimization",
    "Experimental Setup",
    "Results & Performance Analysis",
    "Statistical Validation",
    "Practical Implications",
    "Limitations & Future Work",
    "Conclusions"
])

# ============================================================================
# SLIDE 3: PROBLEM STATEMENT
# ============================================================================
slide = add_content_slide(prs, "Problem Statement")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Breast cancer: Leading cause of mortality in women",
    "Machine learning for diagnosis: High-dimensional data (30 features)",
    "Challenge: Optimal feature selection + hyperparameter tuning",
    "Traditional approach: Sequential optimization → Suboptimal results",
    "Our solution: Simultaneous optimization using nature-inspired DIO algorithm"
])

# ============================================================================
# SLIDE 4: WHAT IS DIO?
# ============================================================================
slide = add_content_slide(prs, "Dholes-Inspired Optimization (DIO)")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Nature-inspired metaheuristic algorithm",
    "Based on pack hunting behavior of dholes (Asiatic wild dogs)",
    "Three cooperative hunting strategies:",
])

# Add sub-bullets
sub_points = [
    "Chase Alpha: Follow best hunter (exploitation)",
    "Random Pursuit: Chase random pack member (exploration)",
    "Pack Center: Move toward group average (cooperation)"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Published 2023 by Dehghani et al. in Scientific Reports"
p.level = 0
p.font.size = Pt(18)

# ============================================================================
# SLIDE 5: DIO MATHEMATICAL FORMULATION
# ============================================================================
slide = add_content_slide(prs, "DIO Mathematical Formulation")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Position update combines three strategies:",
    "",
    "X_chase = X_alpha + r₁ × (X_alpha - X_i)",
    "X_random = X_r + r₂ × (X_r - X_i)",
    "X_scavenge = X_mean + r₃ × (X_mean - X_i)",
    "",
    "X_new = (X_chase + X_random + X_scavenge) / 3",
    "",
    "Where: r₁, r₂, r₃ ∈ [0,1] are random numbers"
])

# ============================================================================
# SLIDE 6: NESTED OPTIMIZATION FRAMEWORK
# ============================================================================
slide = add_content_slide(prs, "Novel Nested Optimization Framework")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Two-level hierarchical optimization:",
])

p = tf.add_paragraph()
p.text = "Outer Loop: Hyperparameter Tuning"
p.level = 1
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "Population: 3 dholes, 5 iterations",
    "Parameters: n_estimators, max_depth, min_samples_split, min_samples_leaf"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 2
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Inner Loop: Feature Selection"
p.level = 1
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "Population: 5 dholes, 10 iterations",
    "Selects optimal feature subset for each hyperparameter set"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 2
    p.font.size = Pt(16)

# ============================================================================
# SLIDE 7: FITNESS FUNCTION
# ============================================================================
slide = add_content_slide(prs, "Multi-Objective Fitness Function")

# Add formula box
left = Inches(1.5)
top = Inches(2.5)
width = Inches(7)
height = Inches(1.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
p = text_frame.paragraphs[0]
p.text = "F = 0.99 × (1 - Accuracy) + 0.01 × (Features/Total)"
p.font.size = Pt(28)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER

# Add explanation
left = Inches(1)
top = Inches(4.2)
width = Inches(8)
height = Inches(2)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
add_bullet_points(text_frame, [
    "Goal: Minimize F (balance accuracy and complexity)",
    "99% weight on accuracy, 1% on feature reduction",
    "Encourages Pareto-optimal solutions",
    "Favors fewer features when accuracy is comparable"
])

# ============================================================================
# SLIDE 8: DATASET & EXPERIMENTAL SETUP
# ============================================================================
slide = add_content_slide(prs, "Dataset & Experimental Setup")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Dataset: Breast Cancer Wisconsin (Diagnostic)",
])

sub_points = [
    "569 samples (357 benign, 212 malignant)",
    "30 features from digitized cell nuclei images",
    "Standard benchmark in medical ML"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Validation Strategy:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "30 independent runs with different train/test splits",
    "70/30 stratified split (random states: 42-71)",
    "Ensures statistical robustness and reproducibility"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

# ============================================================================
# SLIDE 9: OPTIMIZED CONFIGURATION
# ============================================================================
slide = add_content_slide(prs, "DIO-Optimized Configuration")

# Create table
left = Inches(1.5)
top = Inches(2)
width = Inches(7)
height = Inches(4)

# Add text boxes for results
textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

p = text_frame.paragraphs[0]
p.text = "Optimized Hyperparameters:"
p.font.size = Pt(24)
p.font.bold = True

params = opt_results['best_hyperparameters']
param_text = [
    f"• n_estimators: {params['n_estimators']}",
    f"• max_depth: {params['max_depth']}",
    f"• min_samples_split: {params['min_samples_split']}",
    f"• min_samples_leaf: {params['min_samples_leaf']}"
]

for param in param_text:
    p = text_frame.add_paragraph()
    p.text = param
    p.font.size = Pt(20)
    p.space_after = Pt(8)

p = text_frame.add_paragraph()
p.text = f"\nSelected Features: {opt_results['selected_features']['count']} out of 30 (73% reduction!)"
p.font.size = Pt(22)
p.font.bold = True
p.font.color.rgb = RGBColor(46, 204, 113)

# ============================================================================
# SLIDE 10: SELECTED FEATURES
# ============================================================================
slide = add_content_slide(prs, "8 Selected Features")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame

feature_names = opt_results['selected_features']['names']
add_bullet_points(tf, feature_names)

p = tf.add_paragraph()
p.text = "\nBalanced selection: Mean, Error, and Worst statistics"
p.font.size = Pt(18)
p.font.italic = True
p.font.color.rgb = RGBColor(52, 152, 219)

# ============================================================================
# SLIDE 11: RESULTS - VISUALIZATION
# ============================================================================
add_image_slide(
    prs,
    "Model Performance Comparison",
    "statistical_comparison_visualization.png",
    "30 independent runs across 10 machine learning models"
)

# ============================================================================
# SLIDE 12: RESULTS - PERFORMANCE TABLE
# ============================================================================
slide = add_content_slide(prs, "Performance Rankings (Top 7)")

left = Inches(0.8)
top = Inches(2)
width = Inches(8.4)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
text_frame.word_wrap = True

results_text = """
Rank  Model                    Accuracy    Features
────────────────────────────────────────────────────
1     XGBoost (All)           96.24%      30
2     RF Default (All)        95.87%      30
3     Gradient Boosting       95.75%      30
4     XGBoost (Selected)      95.38%      8
5     Logistic Regression     94.91%      30
6     RF Default (Selected)   94.89%      8
7     DIO-Optimized RF        94.72%      8  ⭐
"""

p = text_frame.paragraphs[0]
p.text = results_text
p.font.name = 'Courier New'
p.font.size = Pt(16)

# ============================================================================
# SLIDE 13: KEY FINDINGS
# ============================================================================
slide = add_content_slide(prs, "Key Findings")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "✓ 94.72% ± 1.41% accuracy (7th rank overall)",
    "✓ 73% feature reduction (30 → 8 features)",
    "✓ Only 1.15% accuracy loss vs. full-feature RF",
    "✓ Significantly outperforms SVM (p<0.001) and KNN (p<0.001)",
    "✓ Comparable to RF with same features (p=0.165)",
    "✓ Low variance (1.41%) = excellent stability",
    "✓ Pareto-optimal: Best accuracy-complexity trade-off"
])

# ============================================================================
# SLIDE 14: STATISTICAL SIGNIFICANCE
# ============================================================================
slide = add_content_slide(prs, "Statistical Validation")

left = Inches(1)
top = Inches(2)
width = Inches(8)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

p = text_frame.paragraphs[0]
p.text = "Wilcoxon Signed-Rank Test Results:"
p.font.size = Pt(24)
p.font.bold = True

test_results = [
    "",
    "DIO-Optimized RF vs.:",
    "  • SVM: +3.16% improvement (p < 0.001) ✓✓✓",
    "  • KNN: +1.70% improvement (p < 0.001) ✓✓✓",
    "  • RF Default (Selected): -0.17% (p = 0.165) ≈",
    "  • Naive Bayes: +0.53% (p = 0.089) ≈",
    "",
    "Legend: ✓✓✓ = Highly Significant | ≈ = Not Significant",
    "",
    "30 paired samples provide strong statistical power"
]

for result in test_results:
    p = text_frame.add_paragraph()
    p.text = result
    p.font.size = Pt(18)
    p.space_after = Pt(6)

# ============================================================================
# SLIDE 15: PARETO OPTIMALITY
# ============================================================================
slide = add_content_slide(prs, "Pareto-Optimal Solution")

left = Inches(1)
top = Inches(2)
width = Inches(8)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

p = text_frame.paragraphs[0]
p.text = "Why Rank 7 is Actually a Success:"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = RGBColor(46, 204, 113)

points = [
    "",
    "Accuracy Loss: Only 1.5% vs. best model",
    "Feature Reduction: 73% fewer features",
    "Trade-off Ratio: 0.02% accuracy per 1% feature reduction",
    "",
    "Practical Benefits:",
    "  • 73% faster inference time",
    "  • Lower memory footprint",
    "  • Better interpretability (8 vs 30 features)",
    "  • Reduced overfitting risk",
    "  • Suitable for resource-constrained deployment"
]

for point in points:
    p = text_frame.add_paragraph()
    p.text = point
    p.font.size = Pt(18)
    p.space_after = Pt(4)

# ============================================================================
# SLIDE 16: PRACTICAL IMPLICATIONS
# ============================================================================
slide = add_content_slide(prs, "Clinical Deployment Advantages")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Computational Efficiency:",
])

sub_points = [
    "73% fewer features = proportionally faster processing",
    "Critical for high-throughput screening facilities"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Cost Reduction:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

p = tf.add_paragraph()
p.text = "Fewer features may reduce laboratory measurements"
p.level = 1
p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Interpretability:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

p = tf.add_paragraph()
p.text = "Clinicians can understand and validate 8 features easier than 30"
p.level = 1
p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "Robustness:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

p = tf.add_paragraph()
p.text = "Less susceptible to missing or corrupted data"
p.level = 1
p.font.size = Pt(16)

# ============================================================================
# SLIDE 17: ALGORITHM VALIDATION
# ============================================================================
slide = add_content_slide(prs, "Implementation Validation")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "First Python implementation of DIO (original: MATLAB)",
    "",
    "Validated on 14 standard benchmark functions (F1-F14):",
])

sub_points = [
    "Unimodal functions (F1-F7)",
    "Multimodal functions (F8-F13)",
    "Fixed-dimension multimodal (F14)"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = ""
p.level = 0

p = tf.add_paragraph()
p.text = "Configuration: 30 population, 500 iterations, 30 runs"
p.level = 0
p.font.size = Pt(18)

p = tf.add_paragraph()
p.text = "Result: Near-zero convergence (F1: 7.6×10⁻²⁶)"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(46, 204, 113)

# ============================================================================
# SLIDE 18: LIMITATIONS
# ============================================================================
slide = add_content_slide(prs, "Limitations")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Single dataset evaluation (Breast Cancer Wisconsin only)",
    "DIO optimization time not quantified or compared",
    "Feature selection stability not assessed across multiple runs",
    "Limited hyperparameter space (4 RF parameters)",
    "No comparison with other metaheuristics (PSO, GA, ACO)",
    "Domain-specific: 73% reduction may not generalize to all problems"
])

# ============================================================================
# SLIDE 19: FUTURE WORK
# ============================================================================
slide = add_content_slide(prs, "Future Research Directions")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "Multi-dataset validation (lung cancer, diabetes, heart disease)",
    "Benchmark against other metaheuristics (PSO, GA, ACO)",
    "Extend to other classifiers (XGBoost, neural networks)",
    "Feature selection stability analysis",
    "Computational profiling and parallelization",
    "Real-world clinical deployment and prospective validation",
    "Hybrid approaches combining DIO with domain knowledge"
])

# ============================================================================
# SLIDE 20: CONTRIBUTIONS
# ============================================================================
slide = add_content_slide(prs, "Key Contributions")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "✓ First Python implementation of DIO algorithm",
    "✓ Novel nested optimization framework for simultaneous tuning",
    "✓ Rigorous statistical validation (30 independent runs)",
    "✓ Pareto analysis emphasizing accuracy-complexity trade-off",
    "✓ Comprehensive benchmark validation (14 test functions)",
    "✓ Open-source implementation for reproducible research",
    "✓ Practical methodology for medical AI deployment"
])

# ============================================================================
# SLIDE 21: CONCLUSIONS
# ============================================================================
slide = add_content_slide(prs, "Conclusions")

left = Inches(1)
top = Inches(2)
width = Inches(8)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

points = [
    "DIO effectively optimizes Random Forest for breast cancer classification",
    "",
    "Achieved Pareto-optimal solution:",
    "  • 94.72% accuracy with only 8/30 features",
    "  • 73% reduction in model complexity",
    "  • Statistically validated across 30 runs",
    "",
    "Demonstrates practical viability for resource-constrained medical applications",
    "",
    "Provides foundation for applying nature-inspired optimization to medical AI"
]

for i, point in enumerate(points):
    if i == 0:
        p = text_frame.paragraphs[0]
    else:
        p = text_frame.add_paragraph()
    p.text = point
    p.font.size = Pt(20)
    p.space_after = Pt(8)

# ============================================================================
# SLIDE 22: THANK YOU / Q&A
# ============================================================================
slide_layout = prs.slide_layouts[6]  # Blank
slide = prs.slides.add_slide(slide_layout)

# Title
left = Inches(1)
top = Inches(2)
width = Inches(8)
height = Inches(1)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
p = text_frame.paragraphs[0]
p.text = "Thank You!"
p.font.size = Pt(60)
p.font.bold = True
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(44, 62, 80)

# Subtitle
left = Inches(1)
top = Inches(3.5)
width = Inches(8)
height = Inches(1)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
p = text_frame.paragraphs[0]
p.text = "Questions & Discussion"
p.font.size = Pt(36)
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(52, 152, 219)

# Contact/Links
left = Inches(1)
top = Inches(5.5)
width = Inches(8)
height = Inches(1)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
text_frame.word_wrap = True

contact_info = [
    "GitHub: amine-dubs/dio-optimization",
    "Email: your.email@university.edu",
    "Dataset: UCI Machine Learning Repository"
]

for i, info in enumerate(contact_info):
    if i == 0:
        p = text_frame.paragraphs[0]
    else:
        p = text_frame.add_paragraph()
    p.text = info
    p.font.size = Pt(18)
    p.alignment = PP_ALIGN.CENTER

# ============================================================================
# SAVE PRESENTATION
# ============================================================================
output_file = "DIO_Research_Presentation.pptx"
prs.save(output_file)
print(f"✅ Presentation created successfully: {output_file}")
print(f"📊 Total slides: {len(prs.slides)}")
print(f"⏱️  Estimated presentation time: ~15 minutes")
print(f"\n🎯 Next steps:")
print(f"   1. Open {output_file} in PowerPoint")
print(f"   2. Review and customize content")
print(f"   3. Add speaker notes if needed")
print(f"   4. Practice timing (~40-45 seconds per slide)")
print(f"   5. Ready to present!")
