#!/usr/bin/env python3
"""
Portfolio Verification Script
Checks that all required files are present and properly structured.
"""

from pathlib import Path
from typing import List, Tuple

class PortfolioVerifier:
    """Verify portfolio completeness."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.errors = []
        self.warnings = []
        self.successes = []
    
    def check_file_exists(self, file_path: str, required: bool = True) -> bool:
        """Check if a file exists."""
        full_path = self.base_path / file_path
        exists = full_path.exists()
        
        if exists:
            size = full_path.stat().st_size
            self.successes.append(f"✅ {file_path} ({size:,} bytes)")
            return True
        else:
            if required:
                self.errors.append(f"❌ MISSING (REQUIRED): {file_path}")
            else:
                self.warnings.append(f"⚠️  MISSING (OPTIONAL): {file_path}")
            return False
    
    def verify_crisp_dm(self):
        """Verify CRISP-DM methodology files."""
        print("\n" + "=" * 80)
        print("CRISP-DM METHODOLOGY VERIFICATION")
        print("=" * 80)
        
        files = [
            ("crisp_dm/CRISP_DM.ipynb", True),
            ("crisp_dm/README.md", True),
            ("crisp_dm/src/data_loader.py", True),
            ("crisp_dm/src/feature_engineering.py", True),
            ("crisp_dm/src/modeling.py", True),
            ("crisp_dm/deployment/app.py", True),
            ("crisp_dm/tests/test_leakage.py", True),
        ]
        
        for file_path, required in files:
            self.check_file_exists(file_path, required)
        
        # Check for critique files (may have timestamps)
        critique_dir = self.base_path / "crisp_dm" / "prompts" / "executed"
        if critique_dir.exists():
            critique_files = list(critique_dir.glob("*critique*.md"))
            if len(critique_files) >= 2:
                self.successes.append(f"✅ crisp_dm/prompts/executed/ ({len(critique_files)} critique files)")
            else:
                self.warnings.append(f"⚠️  Only {len(critique_files)} critique files found (expected 2+)")
    
    def verify_semma(self):
        """Verify SEMMA methodology files."""
        print("\n" + "=" * 80)
        print("SEMMA METHODOLOGY VERIFICATION")
        print("=" * 80)
        
        files = [
            ("semma/SEMMA.ipynb", True),
            ("semma/README.md", True),
            ("semma/requirements.txt", True),
            ("semma/src/data_loader.py", True),
            ("semma/src/preprocessing.py", True),
            ("semma/tests/test_data_quality.py", True),
            ("semma/prompts/executed/semma_critique_kozyrkov.md", True),
        ]
        
        for file_path, required in files:
            self.check_file_exists(file_path, required)
    
    def verify_kdd(self):
        """Verify KDD methodology files."""
        print("\n" + "=" * 80)
        print("KDD METHODOLOGY VERIFICATION")
        print("=" * 80)
        
        files = [
            ("kdd/KDD.ipynb", True),
            ("kdd/README.md", True),
            ("kdd/requirements.txt", True),
            ("kdd/src/data_loader.py", True),
            ("kdd/tests/test_data_pipeline.py", True),
            ("kdd/prompts/executed/kdd_critique_denning.md", True),
        ]
        
        for file_path, required in files:
            self.check_file_exists(file_path, required)
    
    def verify_root(self):
        """Verify root portfolio files."""
        print("\n" + "=" * 80)
        print("ROOT PORTFOLIO VERIFICATION")
        print("=" * 80)
        
        files = [
            ("README.md", True),
            ("requirements.txt", True),
            ("Dockerfile", True),
            (".gitignore", True),
            ("PORTFOLIO_SUMMARY.md", True),
            ("generate_notebooks.py", True),
        ]
        
        for file_path, required in files:
            self.check_file_exists(file_path, required)
    
    def print_summary(self):
        """Print verification summary."""
        print("\n" + "=" * 80)
        print("VERIFICATION SUMMARY")
        print("=" * 80)
        
        print(f"\n✅ Successes: {len(self.successes)}")
        print(f"⚠️  Warnings: {len(self.warnings)}")
        print(f"❌ Errors: {len(self.errors)}")
        
        if self.errors:
            print("\n" + "=" * 80)
            print("ERRORS (REQUIRED FILES MISSING)")
            print("=" * 80)
            for error in self.errors:
                print(error)
        
        if self.warnings:
            print("\n" + "=" * 80)
            print("WARNINGS (OPTIONAL FILES MISSING)")
            print("=" * 80)
            for warning in self.warnings:
                print(warning)
        
        # Overall status
        print("\n" + "=" * 80)
        if self.errors:
            print("❌ VERIFICATION FAILED")
            print(f"   {len(self.errors)} required files missing")
        else:
            print("✅ VERIFICATION PASSED")
            print("   All required files present")
        print("=" * 80)
    
    def count_lines_of_code(self):
        """Count total lines of code in the portfolio."""
        print("\n" + "=" * 80)
        print("CODE STATISTICS")
        print("=" * 80)
        
        total_lines = 0
        total_files = 0
        
        for py_file in self.base_path.rglob("*.py"):
            # Skip __pycache__ and .venv
            if "__pycache__" in str(py_file) or ".venv" in str(py_file):
                continue
            
            try:
                with open(py_file, 'r') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    total_files += 1
                    print(f"  {py_file.relative_to(self.base_path)}: {lines} lines")
            except:
                pass
        
        print(f"\nTotal Python files: {total_files}")
        print(f"Total lines of code: {total_lines:,}")
        
        # Count notebooks
        nb_count = len(list(self.base_path.rglob("*.ipynb")))
        print(f"Total notebooks: {nb_count}")
        
        # Count markdown files
        md_files = [f for f in self.base_path.rglob("*.md") if "node_modules" not in str(f)]
        md_count = len(md_files)
        md_lines = sum(len(open(f, 'r').readlines()) for f in md_files)
        print(f"Total markdown files: {md_count} ({md_lines:,} lines)")
    
    def verify_all(self):
        """Run all verifications."""
        print("\n" + "🔍" * 40)
        print("DATA MINING METHODOLOGIES PORTFOLIO VERIFICATION")
        print("🔍" * 40)
        
        self.verify_root()
        self.verify_crisp_dm()
        self.verify_semma()
        self.verify_kdd()
        self.count_lines_of_code()
        self.print_summary()


def main():
    """Main verification function."""
    verifier = PortfolioVerifier(".")
    verifier.verify_all()


if __name__ == "__main__":
    main()
