import pandas as pd
import numpy as np
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AmazonDatasetPreprocessor:
    """
    Preprocessor for Amazon Sales Dataset
    Handles cleaning, transformation, and preparation for SQL-like queries
    """
    
    def __init__(self, csv_path: str):
        """Initialize with the CSV file path."""
        self.csv_path = csv_path
        self.df = None
        self.original_df = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV file."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path)
        self.original_df = self.df.copy()
        print(f"✓ Loaded {len(self.df)} rows and {len(self.df.columns)} columns")
        return self.df
    
    def clean_price_columns(self):
        """Clean and convert price columns to numeric format."""
        print("\n🧹 Cleaning price columns...")
        
        price_columns = ['discounted_price', 'actual_price']
        
        for col in price_columns:
            if col in self.df.columns:
                # Remove currency symbols and commas
                self.df[col] = self.df[col].astype(str).str.replace('₹', '', regex=False)
                self.df[col] = self.df[col].str.replace(',', '', regex=False)
                self.df[col] = self.df[col].str.strip()
                
                # Convert to numeric
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                
                print(f"  ✓ Cleaned {col}: {self.df[col].notna().sum()} valid values")
    
    def clean_discount_column(self):
        """Clean and convert discount percentage to numeric."""
        print("\n🧹 Cleaning discount column...")
        
        if 'discount_percentage' in self.df.columns:
            # Remove percentage symbol
            self.df['discount_percentage'] = self.df['discount_percentage'].astype(str).str.replace('%', '', regex=False)
            self.df['discount_percentage'] = pd.to_numeric(self.df['discount_percentage'], errors='coerce')
            
            print(f"  ✓ Cleaned discount_percentage: {self.df['discount_percentage'].notna().sum()} valid values")
    
    def clean_rating_columns(self):
        """Clean and convert rating columns to numeric."""
        print("\n🧹 Cleaning rating columns...")
        
        if 'rating' in self.df.columns:
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')
            print(f"  ✓ Cleaned rating: {self.df['rating'].notna().sum()} valid values")
        
        if 'rating_count' in self.df.columns:
            # Remove commas from rating count
            self.df['rating_count'] = self.df['rating_count'].astype(str).str.replace(',', '', regex=False)
            self.df['rating_count'] = pd.to_numeric(self.df['rating_count'], errors='coerce')
            print(f"  ✓ Cleaned rating_count: {self.df['rating_count'].notna().sum()} valid values")
    
    def extract_category_hierarchy(self):
        """Extract main category and subcategories from category column."""
        print("\n🧹 Extracting category hierarchy...")
        
        if 'category' in self.df.columns:
            # Split by pipe symbol
            categories_split = self.df['category'].str.split('|', expand=True)
            
            self.df['main_category'] = categories_split[0] if 0 in categories_split.columns else None
            self.df['sub_category_1'] = categories_split[1] if 1 in categories_split.columns else None
            self.df['sub_category_2'] = categories_split[2] if 2 in categories_split.columns else None
            self.df['sub_category_3'] = categories_split[3] if 3 in categories_split.columns else None
            
            print(f"  ✓ Extracted main_category: {self.df['main_category'].notna().sum()} values")
            print(f"  ✓ Extracted sub_category_1: {self.df['sub_category_1'].notna().sum()} values")
    
    def clean_text_columns(self):
        """Clean text columns by removing extra whitespace."""
        print("\n🧹 Cleaning text columns...")
        
        text_columns = ['product_name', 'review_title', 'review_content', 'user_name']
        
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                # Replace multiple spaces with single space
                self.df[col] = self.df[col].str.replace(r'\s+', ' ', regex=True)
        
        print(f"  ✓ Cleaned {len([c for c in text_columns if c in self.df.columns])} text columns")
    
    def create_derived_features(self):
        """Create useful derived features for analysis."""
        print("\n🔧 Creating derived features...")
        
        # Calculate savings amount
        if 'actual_price' in self.df.columns and 'discounted_price' in self.df.columns:
            self.df['savings_amount'] = self.df['actual_price'] - self.df['discounted_price']
            print(f"  ✓ Created savings_amount")
        
        # Create price ranges
        if 'discounted_price' in self.df.columns:
            self.df['price_range'] = pd.cut(
                self.df['discounted_price'], 
                bins=[0, 200, 500, 1000, 2000, float('inf')],
                labels=['Budget (<₹200)', 'Economy (₹200-500)', 'Mid-range (₹500-1000)', 
                        'Premium (₹1000-2000)', 'Luxury (>₹2000)']
            )
            print(f"  ✓ Created price_range")
        
        # Create rating categories
        if 'rating' in self.df.columns:
            self.df['rating_category'] = pd.cut(
                self.df['rating'],
                bins=[0, 2.5, 3.5, 4.0, 4.5, 5.0],
                labels=['Poor', 'Below Average', 'Average', 'Good', 'Excellent']
            )
            print(f"  ✓ Created rating_category")
        
        # Calculate review length
        if 'review_content' in self.df.columns:
            self.df['review_length'] = self.df['review_content'].astype(str).str.len()
            print(f"  ✓ Created review_length")
        
        # Extract brand from product name (first word usually)
        if 'product_name' in self.df.columns:
            self.df['brand'] = self.df['product_name'].str.split().str[0]
            print(f"  ✓ Created brand")
    
    def handle_missing_values(self):
        """Handle missing values in the dataset."""
        print("\n🔧 Handling missing values...")
        
        # Report missing values
        missing_counts = self.df.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0]
        
        if len(missing_counts) > 0:
            print("  Missing values found:")
            for col, count in missing_counts.items():
                pct = (count / len(self.df)) * 100
                print(f"    - {col}: {count} ({pct:.2f}%)")
        else:
            print("  ✓ No missing values found")
    
    def remove_duplicates(self):
        """Remove duplicate rows."""
        print("\n🔧 Removing duplicates...")
        
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates()
        removed_count = initial_count - len(self.df)
        
        print(f"  ✓ Removed {removed_count} duplicate rows")
    
    def get_data_summary(self) -> Dict:
        """Get a comprehensive summary of the dataset."""
        summary = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'missing_values': self.df.isnull().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns)
        }
        
        # Add statistics for numeric columns
        numeric_stats = {}
        for col in summary['numeric_columns']:
            numeric_stats[col] = {
                'min': float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                'max': float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                'mean': float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                'median': float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None
            }
        summary['numeric_stats'] = numeric_stats
        
        return summary
    
    def preprocess_full(self) -> pd.DataFrame:
        """Run all preprocessing steps."""
        print("=" * 60)
        print("STARTING FULL PREPROCESSING PIPELINE")
        print("=" * 60)
        
        self.load_data()
        self.clean_price_columns()
        self.clean_discount_column()
        self.clean_rating_columns()
        self.extract_category_hierarchy()
        self.clean_text_columns()
        self.create_derived_features()
        self.remove_duplicates()
        self.handle_missing_values()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETED")
        print("=" * 60)
        
        return self.df
    
    def save_processed_data(self, output_path: str):
        """Save the processed dataset to a new CSV file."""
        print(f"\n💾 Saving processed data to {output_path}...")
        self.df.to_csv(output_path, index=False)
        print(f"✓ Saved {len(self.df)} rows to {output_path}")
    
    def generate_report(self):
        """Generate a comprehensive preprocessing report."""
        print("\n" + "=" * 60)
        print("PREPROCESSING REPORT")
        print("=" * 60)
        
        summary = self.get_data_summary()
        
        print(f"\n📊 Dataset Overview:")
        print(f"  Total Rows: {summary['total_rows']:,}")
        print(f"  Total Columns: {summary['total_columns']}")
        
        print(f"\n📈 Numeric Columns ({len(summary['numeric_columns'])}):")
        for col in summary['numeric_columns']:
            stats = summary['numeric_stats'].get(col, {})
            if stats.get('mean') is not None:
                print(f"  - {col}:")
                print(f"      Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"      Mean: {stats['mean']:.2f}, Median: {stats['median']:.2f}")
        
        print(f"\n📝 Categorical Columns ({len(summary['categorical_columns'])}):")
        for col in summary['categorical_columns'][:10]:  # Show first 10
            unique_count = self.df[col].nunique()
            print(f"  - {col}: {unique_count} unique values")
        
        print(f"\n🎯 Top Categories:")
        if 'main_category' in self.df.columns:
            top_categories = self.df['main_category'].value_counts().head(5)
            for cat, count in top_categories.items():
                print(f"  - {cat}: {count} products")
        
        print(f"\n⭐ Rating Distribution:")
        if 'rating_category' in self.df.columns:
            rating_dist = self.df['rating_category'].value_counts().sort_index()
            for rating, count in rating_dist.items():
                print(f"  - {rating}: {count} products")
        
        print(f"\n💰 Price Range Distribution:")
        if 'price_range' in self.df.columns:
            price_dist = self.df['price_range'].value_counts().sort_index()
            for price_range, count in price_dist.items():
                print(f"  - {price_range}: {count} products")


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = AmazonDatasetPreprocessor("amazon_sales.csv")
    
    # Run full preprocessing
    processed_df = preprocessor.preprocess_full()
    
    # Generate detailed report
    preprocessor.generate_report()
    
    # Save processed data
    preprocessor.save_processed_data("amazon_sales_processed.csv")
    
    print("\n" + "-" * 100)
    print("✅ PREPROCESSING COMPLETE!")
    print("\nProcessed dataset saved as: amazon_sales_processed.csv")
    print("You can now use this file with the RAG chatbot.")
    
    # Display sample of processed data
    print("\n📋 Sample of processed data (first 3 rows):")
    print(processed_df.head(3)[['product_name', 'discounted_price', 'rating', 
                                  'main_category', 'price_range', 'rating_category']].to_string())