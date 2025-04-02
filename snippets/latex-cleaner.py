import re
import IPython.display as display  # For rendering LaTeX in Jupyter

# Define regex patterns for math elements
MATH_PATTERNS = [
    r"\$\_\{[^}]+\}",    # Subscripts like x$_{a}$
    r"\$\^\{[^}]+\}",    # Superscripts like x$^{2}$
    r"[=+\-*/^]",        # Operators
    r"\b[kK]\b",         # Common variable 'k' in equations
    r"\b[a-zA-Z]_\{[^}]+\}",  # Another subscript format like x_{r}
    r"\b[a-zA-Z]\^\{[^}]+\}", # Superscript format
    r"[αβγδεζηθικλμνξοπρστυφχψω]",  # Greek letters
    r"\[.*?\]",          # Square brackets used mathematically
    r"\b(sin|cos|tan|log|exp|sqrt)\b"  # Common math functions
]

# Define common English words (non-math tokens)
ENGLISH_WORDS = set([
    "the", "is", "and", "for", "with", "this", "to", "on", "it", "that", "in", "we", "if", "not"
])

def classify_text_block(text, math_threshold=0.50):
    """Classifies a text block as <text> or <formula> based on the math ratio."""
    
    tokens = text.split()
    
    # Count math tokens
    math_count = sum(bool(re.search(pattern, token)) for pattern in MATH_PATTERNS for token in tokens)
    
    # Count normal words
    word_count = sum(token.lower() in ENGLISH_WORDS for token in tokens)
    
    # Total tokens (avoid division by zero)
    total_tokens = max(len(tokens), 1)
    
    # Compute math ratio
    math_ratio = math_count / total_tokens
    
    # Determine classification
    classification = "formula" if math_ratio >= math_threshold else "text"
    
    return classification, math_ratio

def batch_process_text_blocks(text_blocks, math_threshold=0.50):
    """Processes a batch of text blocks, reassigning misclassified formulas."""
    results = []
    
    for text in text_blocks:
        classification, ratio = classify_text_block(text, math_threshold)
        if classification == "formula":
            text = normalize_latex(text)  # Normalize for LaTeX
        results.append((classification, ratio, text))
    
    return results

def normalize_latex(text):
    """Converts raw text with LaTeX subscripts/superscripts into proper LaTeX syntax."""
    
    # Fix subscript notation: x$_{a}$ → x_{a}
    text = re.sub(r"\$\_\{([^}]+)\}\$", r"_{\1}", text)
    
    # Fix superscript notation: x$^{2}$ → x^{2}
    text = re.sub(r"\$\^\{([^}]+)\}\$", r"^{\1}", text)
    
    # Convert square brackets to LaTeX-friendly notation
    text = text.replace("[", r"\left[").replace("]", r"\right]")
    
    # Convert Greek letters (β → \beta)
    greek_letters = {
        "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
        "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
        "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
        "ν": r"\nu", "ξ": r"\xi", "ο": r"o", "π": r"\pi", "ρ": r"\rho",
        "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
        "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega"
    }
    
    for greek, latex in greek_letters.items():
        text = text.replace(greek, latex)
    
    return text

def render_latex_in_jupyter(latex_text):
    """Renders LaTeX formula in Jupyter Notebook."""
    display.display(display.Math(latex_text))

# Example batch of misclassified text blocks
text_blocks = [
    "x$_{a}$ ( k ) = K$_{c}$$_{l}$[ x$_{r}$ ( k - 1) + a$_{2}$x$_{r}$ ( k - 2) + a$_{3}$x$_{r}$ ( k - 3)] - β$_{3}$x$_{a}$ ( k - 1)",
    "The system operates in real-time, adjusting the parameters dynamically.",
    "y$_{n}$ = a$_{1}$ y$_{n-1}$ + a$_{2}$ y$_{n-2}$ + b$_{0}$ u$_{n}$ + b$_{1}$ u$_{n-1}$"
]

# Process batch
processed_blocks = batch_process_text_blocks(text_blocks)

# Output results
for i, (classification, ratio, processed_text) in enumerate(processed_blocks):
    print(f"Block {i+1}: <{classification}> (Math Ratio: {ratio:.2%})")
    print(processed_text)
    print("-" * 50)

    # Render LaTeX in Jupyter if classified as formula
    if classification == "formula":
        render_latex_in_jupyter(processed_text)
