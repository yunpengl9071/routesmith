import re

def fix_math(content):
    # Fix specific patterns
    
    # Pattern 1: Beta(\alpha_c, \beta_c) -> Beta($\alpha_c$, $\beta_c$)
    content = re.sub(r'Beta\\(alpha_([a-z])\s*,\s*\\beta_([a-z])', r'Beta($\\alpha_\1$, $\\beta_\2$)', content)
    
    # Pattern 2: \alpha×quality - \beta×cost -> $\alpha\times$quality - $\beta\times$cost
    content = re.sub(r'\\alpha×', r'$\alpha\times$', content)
    content = re.sub(r'\\beta×', r'$\beta\times$', content)
    
    # Pattern 3: \lambda= -> $\lambda$=
    content = re.sub(r'\\lambda=', r'$\lambda$=', content)
    
    # Pattern 4: t(99) = 35.04, p < 0.000001 -> $t(99) = 35.04$, $p < 0.000001$
    content = re.sub(r'(t\([0-9]+\)\s*=\s*[-+]?[0-9]*\.?[0-9]+,\s*p\s*[<>]\s*[0-9]*\.?[0-9]+)', r'$\1$', content)
    
    # Pattern 5: \pm -> $\pm$
    content = re.sub(r'\\pm', r'$\pm$', content)
    
    # Pattern 6: \cdot -> $\cdot$
    content = re.sub(r'\\cdot', r'$\cdot$', content)
    
    # Pattern 7: Fix any remaining \alpha, \beta, \lambda not in math mode
    # This is trickier - need to avoid breaking already fixed ones
    
    return content

with open('routesmith_technical_report.md', 'r', encoding='utf-8') as f:
    content = f.read()

original = content
content = fix_math(content)

# Also fix any \alpha_c, \beta_c not in math mode (manual patterns)
lines = content.split('\n')
for i, line in enumerate(lines):
    if '\\alpha_c' in line and '$\\alpha_c' not in line:
        lines[i] = lines[i].replace('\\alpha_c', '$\\alpha_c$')
    if '\\beta_c' in line and '$\\beta_c' not in line:
        lines[i] = lines[i].replace('\\beta_c', '$\\beta_c$')
    if '\\alpha×' in line:
        lines[i] = lines[i].replace('\\alpha×', '$\\alpha\\times$')
    if '\\beta×' in line:
        lines[i] = lines[i].replace('\\beta×', '$\\beta\\times$')
        
content = '\n'.join(lines)

with open('routesmith_technical_report_fixed.md', 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Original length: {len(original)}")
print(f"Fixed length: {len(content)}")
print("Created routesmith_technical_report_fixed.md")

# Show some examples of fixes
import difflib
diff = difflib.unified_diff(original.split('\n'), content.split('\n'), lineterm='')
count = 0
for line in diff:
    if line.startswith('+') and not line.startswith('+++'):
        print(f"  {line[:100]}")
        count += 1
    if count >= 10:
        break