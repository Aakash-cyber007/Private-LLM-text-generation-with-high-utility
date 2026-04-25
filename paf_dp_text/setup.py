"""
setup.py
========
One-button environment + dataset bootstrap for PaF-DP-Text.

Fixes every known Colab pitfall:
  1. Pins sentence-transformers to 2.7.0 (no torchcodec / FFmpeg)
  2. Strips Windows CRLF from any .sh files
  3. Validates HF_TOKEN before anything tries to download gated models
  4. Downloads & splits the Yelp dataset
  5. Creates required directory tree

Usage (Colab cell):
    !python setup.py --hf_token "hf_..."
or set HF_TOKEN env-var first:
    import os; os.environ['HF_TOKEN'] = 'hf_...'
    !python setup.py
"""

import os, sys, subprocess, argparse, textwrap, getpass

# ── package list — ORDER MATTERS ─────────────────────────────────────────────
PACKAGES = [
    'numpy==2.2.4',
    'pandas==2.2.3',
    'scipy==1.15.2',
    'scikit-learn==1.6.1',
    'tqdm==4.67.1',
    'huggingface_hub==0.35.3',
    'tokenizers>=0.21.0',
    'transformers==4.57.0',
    'accelerate==1.5.2',
    'datasets==3.4.1',
    # ↓ CRITICAL: 2.7.0 = last version without torchcodec dependency
    'sentence-transformers==2.7.0',
    'mauve-text==0.4.0',
    'gdown==5.2.0',
    'matplotlib==3.9.0',
    'seaborn==0.13.2',
    'wordcloud==1.9.3',
]


def run(cmd, check=True):
    print(f'  $ {cmd}')
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode != 0:
        print(f'[ERROR] exit {r.returncode}')
        sys.exit(r.returncode)
    return r.returncode


def banner(msg):
    print('\n' + '─'*64)
    print(f'  {msg}')
    print('─'*64)


# ─────────────────────────────────────────────────────────────
# Step 1 — CRLF fix
# ─────────────────────────────────────────────────────────────
def fix_crlf():
    banner('Step 1 — Fixing Windows line endings')
    count = 0
    for root, _, files in os.walk('.'):
        for f in files:
            if f.endswith('.sh'):
                path = os.path.join(root, f)
                data = open(path, 'rb').read()
                fixed = data.replace(b'\r\n', b'\n').replace(b'\r', b'\n')
                if fixed != data:
                    open(path, 'wb').write(fixed)
                    count += 1
                    print(f'    fixed: {path}')
    print(f'  ✓  {count} file(s) fixed.')


# ─────────────────────────────────────────────────────────────
# Step 2 — Install packages
# ─────────────────────────────────────────────────────────────
def install_packages():
    banner('Step 2 — Installing packages')
    # Remove torchcodec BEFORE installing anything (may already be present)
    run('pip uninstall torchcodec -y 2>/dev/null || true', check=False)
    run(f'{sys.executable} -m pip install --upgrade pip -q')
    for pkg in PACKAGES:
        run(f'{sys.executable} -m pip install "{pkg}" -q')
    # Remove torchcodec again in case a transitive dep pulled it back
    run('pip uninstall torchcodec -y 2>/dev/null || true', check=False)
    print('  ✓  All packages installed.')


# ─────────────────────────────────────────────────────────────
# Step 3 — HF token
# ─────────────────────────────────────────────────────────────
def setup_hf(cli_token=''):
    banner('Step 3 — Hugging Face authentication')
    token = cli_token or os.environ.get('HF_TOKEN', '').strip()
    if not token:
        print(textwrap.dedent("""
            No HF_TOKEN found.
            Get a *read* token at: https://huggingface.co/settings/tokens
            Then either:
              • Pass --hf_token hf_... to this script, OR
              • Set the env var: export HF_TOKEN="hf_..."
        """).strip())
        token = getpass.getpass('\n  Paste token (hidden): ').strip()
    if not token:
        print('  ⚠  No token — skipping HF login (non-gated models only).')
        return
    os.environ['HF_TOKEN'] = token
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print('  ✓  Logged in.')
    except Exception as e:
        print(f'  [ERROR] Login failed: {e}')
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Step 4 — Directories
# ─────────────────────────────────────────────────────────────
def create_dirs():
    banner('Step 4 — Creating directories')
    for d in ['./data/yelp', './results/paf', './results/invink']:
        os.makedirs(d, exist_ok=True)
        print(f'  ✓  {d}')


# ─────────────────────────────────────────────────────────────
# Step 5 — Download & split Yelp
# ─────────────────────────────────────────────────────────────
YELP_CATS = [
    'Arts & Entertainment', 'Bars', 'Beauty & Spas',
    'Event Planning & Services', 'Grocery', 'Health & Medical',
    'Home & Garden', 'Hotels & Travel', 'Restaurants', 'Shopping',
]
YELP_STARS_LABELS = ['1.0', '2.0', '3.0', '4.0', '5.0']


def prepare_yelp():
    banner('Step 5 — Preparing Yelp dataset')
    import pandas as pd, numpy as np

    train_csv = './data/yelp/train.csv'

    # Download if not present
    if not os.path.exists(train_csv):
        print('  Downloading Yelp dataset …')
        run('gdown https://drive.google.com/drive/folders/'
            '1vetnesv9xx0uMYQFcrsEwlG7J9j-zeCT -O ./data/ --folder',
            check=False)
        # Fallback: tiny synthetic placeholder for smoke-testing
        if not os.path.exists(train_csv):
            print('  ⚠  Could not download Yelp. Creating synthetic placeholder.')
            _make_synthetic_yelp(train_csv)

    df = pd.read_csv(train_csv)
    # normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Robust label detection
    cat_col   = next((c for c in df.columns if 'label1' in c or 'category' in c), None)
    star_col  = next((c for c in df.columns if 'label2' in c or 'star' in c or 'score' in c), None)
    text_col  = next((c for c in df.columns if 'text' in c), 'text')

    if cat_col is None or star_col is None:
        # No labels — create dummy labels from row index for smoke-testing
        n_cats, n_stars = len(YELP_CATS), len(YELP_STARS_LABELS)
        df['label1'] = [f'Business Category: {YELP_CATS[i % n_cats]}'
                        for i in range(len(df))]
        df['label2'] = [f'Review Stars: {YELP_STARS_LABELS[i % n_stars]}.0'
                        for i in range(len(df))]
        cat_col, star_col = 'label1', 'label2'

    cat_labels  = sorted(df[cat_col].unique())
    star_labels = sorted(df[star_col].unique())

    written = 0
    for i, cat in enumerate(cat_labels[:10]):
        for j, star in enumerate(star_labels[:5]):
            sub = df[(df[cat_col] == cat) & (df[star_col] == star)].copy()
            sub['text'] = sub[text_col].fillna('').apply(
                lambda s: ' '.join(str(s).split()))
            out = f'./data/yelp/train_{i}_{j}.csv'
            sub[['text', cat_col, star_col]].to_csv(out, index=False)
            written += 1

    print(f'  ✓  Split into {written} category×star CSV files.')


def _make_synthetic_yelp(path):
    """Minimal synthetic data for smoke-testing without downloading."""
    import pandas as pd
    rows = []
    for i, cat in enumerate(YELP_CATS):
        for j, star in enumerate(YELP_STARS_LABELS):
            for k in range(50):
                rows.append({
                    'text': (f'This is a sample {star}-star review for a '
                             f'{cat} business. Item {k}.'),
                    'label1': f'Business Category: {cat}',
                    'label2': f'Review Stars: {star}',
                })
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  ✓  Synthetic Yelp placeholder written to {path}')


# ─────────────────────────────────────────────────────────────
# Step 6 — Smoke test
# ─────────────────────────────────────────────────────────────
def smoke_test():
    banner('Step 6 — Smoke test')
    tests = [
        'import torch; print("torch", torch.__version__)',
        'import transformers; print("transformers", transformers.__version__)',
        'import sentence_transformers; print("sentence-transformers", sentence_transformers.__version__)',
        'import mauve; print("mauve OK")',
        'import matplotlib; print("matplotlib", matplotlib.__version__)',
    ]
    ok = True
    for t in tests:
        ret = run(f'{sys.executable} -c "{t}"', check=False)
        if ret != 0:
            print(f'  [FAIL] {t[:60]}')
            ok = False
    # Verify torchcodec is gone
    ret = run(f'{sys.executable} -c "import torchcodec"', check=False)
    if ret == 0:
        run('pip uninstall torchcodec -y', check=False)
    else:
        print('  ✓  torchcodec absent (good)')
    if not ok:
        print('\n  ⚠  Some imports failed — check output above.')
    else:
        print('\n  ✓  All smoke tests passed!')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--hf_token',       default='')
    p.add_argument('--skip_install',   action='store_true')
    p.add_argument('--skip_hf',        action='store_true')
    p.add_argument('--skip_dataset',   action='store_true')
    args = p.parse_args()

    print('\n' + '█'*64)
    print('█  PaF-DP-Text  ·  One-button Setup')
    print('█  Target: Google Colab T4 (15 GB)')
    print('█'*64)

    fix_crlf()
    if not args.skip_install:
        install_packages()
    if not args.skip_hf:
        setup_hf(args.hf_token)
    create_dirs()
    if not args.skip_dataset:
        prepare_yelp()
    smoke_test()

    banner('Setup complete!')
    print(textwrap.dedent("""
        Next: open  PaF_vs_InvisibleInk.ipynb  and run all cells.
        It will:
          1. Run both PaF and InvisibleInk generation automatically
          2. Compute MAUVE, perplexity, and other utility metrics
          3. Display rich side-by-side text comparisons and plots
    """).strip())


if __name__ == '__main__':
    main()
