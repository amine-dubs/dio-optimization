"""
PowerPoint Presentation Generator for DIO Research Project
Creates a 15-minute presentation ready for export
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.dml.color import RGBColor
import json
import os

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)

# Load optimization results from parent directory
with open(os.path.join(parent_dir, 'optimization_results.json'), 'r') as f:
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
# SLIDE 12: RESULTS - THREE APPROACHES COMPARISON
# ============================================================================
slide = add_content_slide(prs, "Three Optimization Approaches Validated")

left = Inches(0.5)
top = Inches(2)
width = Inches(9)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
text_frame.word_wrap = True

results_text = """
Approach              Accuracy        Features  Opt.Time  Rank
───────────────────────────────────────────────────────────────
DIO-XGBoost          96.34% ±1.23%   17/30     54 sec    #1 🏆
(Single-Split)       (43% reduction)

DIO-RF-CV            96.26% ±1.33%   6/30      7.9 hrs   #3
(CV-Based)           (80% reduction)

DIO-RF-Single        94.72% ±1.41%   8/30      1 min     #7
(Initial)            (73% reduction)
───────────────────────────────────────────────────────────────

Key Insight: XGBoost achieves BEST OVERALL performance!
"""

p = text_frame.paragraphs[0]
p.text = results_text
p.font.name = 'Courier New'
p.font.size = Pt(16)

# ============================================================================
# SLIDE 13: KEY FINDINGS - UPDATED WITH ALL APPROACHES
# ============================================================================
slide = add_content_slide(prs, "Key Findings: Algorithm-Dependent Optimization")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "🏆 DIO-XGBoost: BEST OVERALL (96.34%, 17 features, Rank #1)",
    "✓ Lowest standard deviation (1.23%) = Most stable",
    "✓ 54-second optimization (526× faster than CV-RF!)",
    "✓ Significantly better than defaults (p=0.0426)",
    "",
    "🥉 DIO-RF-CV: Best Interpretability (96.26%, 6 features, Rank #3)",
    "✓ 80% feature reduction (highest among all)",
    "✓ Significantly better than defaults (p=0.0084)",
    "✓ Fixes optimization overfitting problem",
    "",
    "💡 Critical Discovery: Optimization Overfitting is Algorithm-Dependent",
    "• RF single-split suffered overfitting (100% → 94.72%)",
    "• XGBoost single-split achieved top performance (98.83% → 96.34%)",
    "• Gradient boosting's regularization enables single-split success!"
])

# ============================================================================
# SLIDE 14: CV-BASED OPTIMIZATION VISUALIZATION
# ============================================================================
add_image_slide(
    prs,
    "CV-Based Optimization: Fixing Overfitting",
    os.path.join(parent_dir, "cv_optimization", "statistical_comparison_visualization_cv.png"),
    "96.26% ± 1.33% with 6 features (80% reduction) - Rank #3"
)

# ============================================================================
# SLIDE 15: XGBOOST OPTIMIZATION VISUALIZATION
# ============================================================================
add_image_slide(
    prs,
    "XGBoost Optimization: Best Overall Performance",
    os.path.join(parent_dir, "xgboost_statistical_comparison_visualization.png"),
    "96.34% ± 1.23% with 17 features (43% reduction) - Rank #1 🏆"
)

# ============================================================================
# SLIDE 16: STATISTICAL SIGNIFICANCE - ALL APPROACHES
# ============================================================================
slide = add_content_slide(prs, "Statistical Validation: All Three Approaches")

left = Inches(0.8)
top = Inches(2)
width = Inches(8.4)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

p = text_frame.paragraphs[0]
p.text = "Wilcoxon Signed-Rank Test Results:"
p.font.size = Pt(22)
p.font.bold = True

test_results = [
    "",
    "DIO-XGBoost-Optimized (BEST - Rank #1):",
    "  • vs. XGBoost defaults: p=0.0426 (*) ✓",
    "  • vs. XGBoost (All): p=0.5067 (ns) - equivalent with 43% fewer features!",
    "  • vs. SVM/KNN/NB: p<0.001 (***) ✓✓✓",
    "",
    "DIO-RF-CV-Optimized (Rank #3):",
    "  • vs. RF defaults (6 feat): p=0.0084 (**) ✓✓",
    "  • vs. RF (All): p=0.0553 (ns) - comparable with 80% fewer features!",
    "  • vs. SVM: p<0.001 (***) ✓✓✓",
    "",
    "DIO-RF-Single (Rank #7 - optimization overfitting issue):",
    "  • vs. RF defaults (8 feat): p=0.165 (ns) - generalization problem",
    "  • vs. SVM/KNN: p<0.001 (***) ✓✓✓"
]

for result in test_results:
    p = text_frame.add_paragraph()
    p.text = result
    p.font.size = Pt(15)
    p.space_after = Pt(4)

# ============================================================================
# SLIDE 17: PARETO FRONTIER - THREE MODELS
# ============================================================================
slide = add_content_slide(prs, "Pareto Frontier: Three Validated Solutions")

left = Inches(0.8)
top = Inches(2)
width = Inches(8.4)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

p = text_frame.paragraphs[0]
p.text = "Three Pareto-Optimal Solutions for Different Priorities:"
p.font.size = Pt(24)
p.font.bold = True
p.font.color.rgb = RGBColor(46, 204, 113)

points = [
    "",
    "🏆 Maximum Accuracy: DIO-XGBoost (96.34%, 17 features)",
    "  • Best overall performance across ALL experiments",
    "  • Lowest variance (1.23%) = Most stable predictions",
    "  • Fast optimization (54 seconds)",
    "  • Choose when: Accuracy is paramount",
    "",
    "🎯 Maximum Interpretability: DIO-RF-CV (96.26%, 6 features)",
    "  • Highest feature reduction (80%)",
    "  • Near-maximum accuracy with minimal complexity",
    "  • Choose when: Clinical transparency critical",
    "",
    "⚡ Rapid Prototyping: DIO-RF-Single (94.72%, 8 features)",
    "  • 1-minute optimization time",
    "  • Demonstrates optimization overfitting issue",
    "  • Choose when: Quick baseline needed"
]

for point in points:
    p = text_frame.add_paragraph()
    p.text = point
    p.font.size = Pt(16)
    p.space_after = Pt(4)

# ============================================================================
# SLIDE 18: PRACTICAL IMPLICATIONS - UPDATED
# ============================================================================
slide = add_content_slide(prs, "Clinical Deployment: Choose Your Model")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "🏆 Choose DIO-XGBoost (96.34%, 17 features) if:",
])

sub_points = [
    "Maximum accuracy is critical (e.g., high-stakes screening)",
    "Computational resources available",
    "43% feature reduction still provides efficiency gains"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "🎯 Choose DIO-RF-CV (96.26%, 6 features) if:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "Clinical interpretability paramount (only 6 measurements)",
    "Resource-constrained environments (80% cost reduction)",
    "Near-maximum accuracy (only 0.08% less than XGBoost)"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "⚡ Choose DIO-RF-Single (94.72%, 8 features) if:"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "Rapid prototyping needed (1-minute optimization)",
    "Lower accuracy acceptable for initial screening"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

# ============================================================================
# SLIDE 19: ALGORITHM VALIDATION
# ============================================================================
slide = add_content_slide(prs, "Framework Generalizability Validated")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "DIO Framework Successfully Applied to Multiple Algorithms:",
    "",
    "✓ Random Forest (Bagging ensemble)",
])

sub_points = [
    "Single-split: 94.72% (suffered optimization overfitting)",
    "CV-based: 96.26% (fixed overfitting, rank #3)"
]
for point in sub_points:
    p = tf.add_paragraph()
    p.text = point
    p.level = 1
    p.font.size = Pt(16)

p = tf.add_paragraph()
p.text = "✓ XGBoost (Gradient boosting)"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True

sub_points = [
    "Single-split: 96.34% (best overall, rank #1)",
    "Inherent regularization prevents optimization overfitting!"
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
p.text = "Key Discovery: Algorithm-dependent optimization behavior"
p.level = 0
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(231, 76, 60)

# ============================================================================
# SLIDE 20: LIMITATIONS - UPDATED
# ============================================================================
slide = add_content_slide(prs, "Limitations & Lessons Learned")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "✓ RESOLVED: Optimization overfitting in Random Forest",
    "  → CV-based approach increased accuracy from 94.72% to 96.26%",
    "  → Feature reduction improved from 73% to 80%",
    "",
    "✓ DISCOVERY: Algorithm-dependent optimization behavior",
    "  → XGBoost's regularization enables successful single-split optimization",
    "  → Different algorithms require different validation strategies",
    "",
    "Remaining Limitations:",
    "• Single dataset evaluation (Breast Cancer Wisconsin only)",
    "• Limited hyperparameter spaces (4 RF, 5 XGBoost parameters)",
    "• No comparison with other metaheuristics (PSO, GA, ACO)",
    "• Feature selection stability not assessed across multiple runs"
])

# ============================================================================
# SLIDE 21: FUTURE WORK - UPDATED
# ============================================================================
slide = add_content_slide(prs, "Future Research Directions")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "1. Multi-dataset validation",
    "  → Lung cancer, diabetes, heart disease datasets",
    "  → Verify algorithm-dependent optimization findings",
    "",
    "2. CV-based XGBoost optimization",
    "  → Can 96.34% accuracy be improved further with CV?",
    "  → Compare single-split vs CV for gradient boosting",
    "",
    "3. Benchmark against other metaheuristics (PSO, GA, ACO, GWO)",
    "4. Extend to deep learning (neural architecture search)",
    "5. Feature selection stability analysis across multiple DIO runs",
    "6. Real-world clinical deployment and prospective validation",
    "7. Computational efficiency analysis and parallelization"
])

# ============================================================================
# SLIDE 22: CONTRIBUTIONS - UPDATED
# ============================================================================
slide = add_content_slide(prs, "Key Contributions")
body_shape = slide.placeholders[1]
tf = body_shape.text_frame
add_bullet_points(tf, [
    "✓ First Python implementation of DIO algorithm",
    "✓ Novel nested optimization framework for simultaneous tuning",
    "✓ Three validated optimization approaches with trade-off analysis",
    "✓ Discovery of algorithm-dependent optimization overfitting",
    "✓ Multi-algorithm validation (Random Forest + XGBoost)",
    "✓ Rigorous statistical validation (30 independent runs per approach)",
    "✓ Comprehensive Pareto analysis with deployment recommendations",
    "✓ Open-source implementation for reproducible research",
    "✓ Best-in-class performance: 96.34% accuracy (Rank #1)"
])

# ============================================================================
# SLIDE 23: CONCLUSIONS - UPDATED
# ============================================================================
slide = add_content_slide(prs, "Conclusions")

left = Inches(0.8)
top = Inches(2)
width = Inches(8.4)
height = Inches(4.5)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame

points = [
    "DIO framework successfully optimizes multiple ML algorithms for breast cancer classification",
    "",
    "Three Pareto-optimal solutions achieved:",
    "  🏆 DIO-XGBoost: 96.34% accuracy, 17 features (Rank #1 OVERALL)",
    "  🥉 DIO-RF-CV: 96.26% accuracy, 6 features (Rank #3, best interpretability)",
    "  ⚡ DIO-RF-Single: 94.72% accuracy, 8 features (Rank #7, rapid baseline)",
    "",
    "Key Discovery: Optimization overfitting is algorithm-dependent",
    "  • Gradient boosting's regularization enables single-split success",
    "  • Bagging ensembles benefit from CV-based optimization",
    "",
    "Provides validated framework for medical AI with flexible deployment options"
]

for i, point in enumerate(points):
    if i == 0:
        p = text_frame.paragraphs[0]
    else:
        p = text_frame.add_paragraph()
    p.text = point
    p.font.size = Pt(18)
    p.space_after = Pt(6)

# ============================================================================
# SLIDE 24: THANK YOU / Q&A
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

# Subtitle with achievement
left = Inches(1)
top = Inches(3.2)
width = Inches(8)
height = Inches(1.2)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
p = text_frame.paragraphs[0]
p.text = "🏆 Best Overall Performance: 96.34% Accuracy"
p.font.size = Pt(28)
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(46, 204, 113)
p.font.bold = True

p = text_frame.add_paragraph()
p.text = "Questions & Discussion"
p.font.size = Pt(32)
p.alignment = PP_ALIGN.CENTER
p.font.color.rgb = RGBColor(52, 152, 219)
p.space_before = Pt(12)

# Contact/Links
left = Inches(1)
top = Inches(5.8)
width = Inches(8)
height = Inches(1)

textbox = slide.shapes.add_textbox(left, top, width, height)
text_frame = textbox.text_frame
text_frame.word_wrap = True

contact_info = [
    "GitHub: amine-dubs/dio-optimization",
    "Results: 3 approaches validated | 90+ independent runs",
    "Dataset: UCI Machine Learning Repository (Wisconsin Breast Cancer)"
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
output_file = os.path.join(script_dir, "DIO_Research_Presentation.pptx")
prs.save(output_file)
print(f"✅ Presentation created successfully: {output_file}")
print(f"📊 Total slides: {len(prs.slides)}")
print(f"⏱️  Estimated presentation time: ~18 minutes")
print(f"\n🎯 Highlights:")
print(f"   🏆 DIO-XGBoost: 96.34% accuracy (Rank #1 OVERALL)")
print(f"   🥉 DIO-RF-CV: 96.26% accuracy, 6 features (Rank #3)")
print(f"   💡 Algorithm-dependent optimization overfitting discovered")
print(f"\n📝 Next steps:")
print(f"   1. Open {output_file} in PowerPoint")
print(f"   2. Review three-approach comparison")
print(f"   3. Add speaker notes if needed")
print(f"   4. Practice timing (~45 seconds per slide)")
print(f"   5. Ready to present with complete results!")
