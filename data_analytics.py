"""
Hybrid ML + Gen AI Data Analytics System
Combines traditional ML analysis with LLM-powered insights and chart generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
from typing import Dict, List, Tuple, Any
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class DataAnalyzer:
    """
    Traditional ML-based data analysis for e-commerce data
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = StandardScaler()

    def price_analysis(self) -> Dict[str, Any]:
        """
        Comprehensive price analysis including statistics and trends
        """
        analysis = {
            'statistics': {
                'mean_price': self.df['discounted_price'].mean(),
                'median_price': self.df['discounted_price'].median(),
                'std_price': self.df['discounted_price'].std(),
                'min_price': self.df['discounted_price'].min(),
                'max_price': self.df['discounted_price'].max(),
                'q25': self.df['discounted_price'].quantile(0.25),
                'q75': self.df['discounted_price'].quantile(0.75),
            },
            'price_ranges': self.df['price_range'].value_counts().to_dict() if 'price_range' in self.df.columns else {},
            'discount_stats': {
                'avg_discount': self.df['discount_percentage'].mean(),
                'median_discount': self.df['discount_percentage'].median(),
                'max_discount': self.df['discount_percentage'].max(),
            }
        }

        return analysis

    def sales_trends_analysis(self) -> Dict[str, Any]:
        """
        Analyze sales trends by category, price range, and ratings
        """
        trends = {}

        # Category-wise analysis
        if 'main_category' in self.df.columns:
            category_stats = self.df.groupby('main_category').agg({
                'discounted_price': ['mean', 'count'],
                'rating': 'mean',
                'rating_count': 'sum'
            }).round(2)

            trends['by_category'] = category_stats.to_dict()

        # Price range analysis
        if 'price_range' in self.df.columns:
            price_range_stats = self.df.groupby('price_range').agg({
                'product_id': 'count',
                'rating': 'mean',
                'discount_percentage': 'mean'
            }).round(2)

            trends['by_price_range'] = price_range_stats.to_dict()

        # Top performing products
        trends['top_rated'] = self.df.nlargest(10, 'rating')[
            ['product_name', 'rating', 'rating_count', 'discounted_price']
        ].to_dict('records')

        # High demand products (by rating count)
        trends['high_demand'] = self.df.nlargest(10, 'rating_count')[
            ['product_name', 'rating', 'rating_count', 'discounted_price']
        ].to_dict('records')

        return trends

    def customer_behavior_analysis(self) -> Dict[str, Any]:
        """
        Analyze customer preferences and behavior patterns
        """
        behavior = {}

        # Price vs Rating correlation
        if 'discounted_price' in self.df.columns and 'rating' in self.df.columns:
            correlation = self.df['discounted_price'].corr(self.df['rating'])
            behavior['price_rating_correlation'] = round(correlation, 3)

        # Rating distribution
        behavior['rating_distribution'] = self.df['rating_category'].value_counts().to_dict() if 'rating_category' in self.df.columns else {}

        # Popular price ranges
        if 'price_range' in self.df.columns:
            popular_ranges = self.df['price_range'].value_counts().head(3)
            behavior['popular_price_ranges'] = popular_ranges.to_dict()

        return behavior

    def perform_clustering(self, n_clusters: int = 4) -> Dict[str, Any]:
        """
        Perform K-Means clustering on products based on price, rating, and discount
        """
        # Select features for clustering
        features = ['discounted_price', 'rating', 'discount_percentage']
        X = self.df[features].dropna()

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Add cluster labels to a copy of the data
        clustered_df = self.df.loc[X.index].copy()
        clustered_df['cluster'] = clusters

        # Analyze each cluster
        cluster_profiles = []
        for i in range(n_clusters):
            cluster_data = clustered_df[clustered_df['cluster'] == i]
            profile = {
                'cluster_id': i,
                'size': len(cluster_data),
                'avg_price': cluster_data['discounted_price'].mean(),
                'avg_rating': cluster_data['rating'].mean(),
                'avg_discount': cluster_data['discount_percentage'].mean(),
                'characteristics': self._characterize_cluster(cluster_data)
            }
            cluster_profiles.append(profile)

        return {
            'n_clusters': n_clusters,
            'cluster_profiles': cluster_profiles,
            'clustered_data': clustered_df[['product_name', 'discounted_price', 'rating', 'cluster']].head(20).to_dict('records')
        }

    def _characterize_cluster(self, cluster_df: pd.DataFrame) -> str:
        """Generate human-readable cluster characteristics"""
        avg_price = cluster_df['discounted_price'].mean()
        avg_rating = cluster_df['rating'].mean()

        if avg_price < 500 and avg_rating > 4.0:
            return "Budget-Friendly & High Quality"
        elif avg_price < 500:
            return "Budget Products"
        elif avg_price > 2000 and avg_rating > 4.0:
            return "Premium & High Quality"
        elif avg_price > 2000:
            return "Premium Products"
        elif avg_rating > 4.0:
            return "Mid-Range & High Quality"
        else:
            return "Mid-Range Products"


class ChartGenerator:
    """
    Generate insightful charts and visualizations using matplotlib and plotly
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def generate_price_distribution_chart(self) -> Tuple[str, str]:
        """Generate price distribution histogram"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(self.df['discounted_price'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Price (₹)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Price Distribution')
        ax1.axvline(self.df['discounted_price'].mean(), color='red', linestyle='--', label='Mean')
        ax1.axvline(self.df['discounted_price'].median(), color='green', linestyle='--', label='Median')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot
        ax2.boxplot(self.df['discounted_price'], vert=True, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_ylabel('Price (₹)')
        ax2.set_title('Price Box Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to base64
        img_str = self._fig_to_base64(fig)
        plt.close(fig)

        description = f"""Price Distribution Analysis:
- Average Price: ₹{self.df['discounted_price'].mean():.2f}
- Median Price: ₹{self.df['discounted_price'].median():.2f}
- Price Range: ₹{self.df['discounted_price'].min():.2f} - ₹{self.df['discounted_price'].max():.2f}
- Standard Deviation: ₹{self.df['discounted_price'].std():.2f}"""

        return img_str, description

    def generate_sales_by_category_chart(self) -> Tuple[str, str]:
        """Generate category-wise sales analysis chart"""
        if 'main_category' not in self.df.columns:
            return None, "Category data not available"

        # Get top 10 categories
        top_categories = self.df['main_category'].value_counts().head(10)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart - Product count by category
        ax1.barh(top_categories.index, top_categories.values, color='coral')
        ax1.set_xlabel('Number of Products')
        ax1.set_title('Top 10 Categories by Product Count')
        ax1.grid(True, alpha=0.3, axis='x')

        # Average price by category
        avg_prices = self.df.groupby('main_category')['discounted_price'].mean().sort_values(ascending=False).head(10)
        ax2.barh(avg_prices.index, avg_prices.values, color='teal')
        ax2.set_xlabel('Average Price (₹)')
        ax2.set_title('Top 10 Categories by Average Price')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        img_str = self._fig_to_base64(fig)
        plt.close(fig)

        description = f"""Category Analysis:
- Total Categories: {self.df['main_category'].nunique()}
- Largest Category: {top_categories.index[0]} ({top_categories.values[0]} products)
- Highest Avg Price Category: {avg_prices.index[0]} (₹{avg_prices.values[0]:.2f})"""

        return img_str, description

    def generate_rating_analysis_chart(self) -> Tuple[str, str]:
        """Generate rating distribution and analysis chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Rating distribution
        ax1.hist(self.df['rating'], bins=20, color='gold', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Rating')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Rating Distribution')
        ax1.axvline(self.df['rating'].mean(), color='red', linestyle='--', label='Mean')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Price vs Rating scatter
        sample_df = self.df.sample(min(1000, len(self.df)))
        ax2.scatter(sample_df['discounted_price'], sample_df['rating'], alpha=0.5, color='purple')
        ax2.set_xlabel('Price (₹)')
        ax2.set_ylabel('Rating')
        ax2.set_title('Price vs Rating Relationship')
        ax2.grid(True, alpha=0.3)

        # Rating count distribution
        ax3.hist(np.log10(self.df['rating_count'] + 1), bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Log10(Rating Count + 1)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Rating Count Distribution (Log Scale)')
        ax3.grid(True, alpha=0.3)

        # Rating categories
        if 'rating_category' in self.df.columns:
            rating_cats = self.df['rating_category'].value_counts()
            ax4.pie(rating_cats.values, labels=rating_cats.index, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Rating Categories Distribution')
        else:
            ax4.axis('off')

        plt.tight_layout()

        img_str = self._fig_to_base64(fig)
        plt.close(fig)

        correlation = self.df['discounted_price'].corr(self.df['rating'])
        description = f"""Rating Analysis:
- Average Rating: {self.df['rating'].mean():.2f}
- Median Rating: {self.df['rating'].median():.2f}
- Price-Rating Correlation: {correlation:.3f}
- Total Reviews: {int(self.df['rating_count'].sum()):,}"""

        return img_str, description

    def generate_discount_analysis_chart(self) -> Tuple[str, str]:
        """Generate discount analysis visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Discount distribution
        ax1.hist(self.df['discount_percentage'], bins=30, color='green', edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Discount Percentage (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Discount Distribution')
        ax1.axvline(self.df['discount_percentage'].mean(), color='red', linestyle='--', label='Mean')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Discount vs Price
        sample_df = self.df.sample(min(1000, len(self.df)))
        ax2.scatter(sample_df['discounted_price'], sample_df['discount_percentage'], alpha=0.5, color='orange')
        ax2.set_xlabel('Discounted Price (₹)')
        ax2.set_ylabel('Discount Percentage (%)')
        ax2.set_title('Price vs Discount Relationship')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        img_str = self._fig_to_base64(fig)
        plt.close(fig)

        description = f"""Discount Analysis:
- Average Discount: {self.df['discount_percentage'].mean():.1f}%
- Median Discount: {self.df['discount_percentage'].median():.1f}%
- Max Discount: {self.df['discount_percentage'].max():.1f}%
- Products with >50% discount: {len(self.df[self.df['discount_percentage'] > 50])}"""

        return img_str, description

    def generate_clustering_chart(self, clustered_data: pd.DataFrame) -> Tuple[str, str]:
        """Generate product clustering visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 2D scatter of clusters (Price vs Rating)
        for cluster_id in clustered_data['cluster'].unique():
            cluster_df = clustered_data[clustered_data['cluster'] == cluster_id]
            ax1.scatter(cluster_df['discounted_price'], cluster_df['rating'],
                       label=f'Cluster {cluster_id}', alpha=0.6, s=50)

        ax1.set_xlabel('Price (₹)')
        ax1.set_ylabel('Rating')
        ax1.set_title('Product Clusters (Price vs Rating)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cluster size distribution
        cluster_sizes = clustered_data['cluster'].value_counts().sort_index()
        ax2.bar(cluster_sizes.index, cluster_sizes.values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(cluster_sizes)])
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Products')
        ax2.set_title('Cluster Size Distribution')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        img_str = self._fig_to_base64(fig)
        plt.close(fig)

        description = f"""Clustering Analysis:
- Number of Clusters: {clustered_data['cluster'].nunique()}
- Products Clustered: {len(clustered_data)}
- Largest Cluster: {cluster_sizes.max()} products"""

        return img_str, description

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64


class HybridAnalyticsEngine:
    """
    Combines traditional ML analytics with LLM-powered interpretation
    """

    def __init__(self, df: pd.DataFrame, gemini_model):
        self.df = df
        self.gemini_model = gemini_model
        self.analyzer = DataAnalyzer(df)
        self.chart_generator = ChartGenerator(df)

    def analyze_and_visualize(self, query: str) -> Dict[str, Any]:
        """
        Main entry point: Analyze data, generate charts, and provide LLM interpretation

        Returns:
            - analysis_type: Type of analysis performed
            - ml_insights: Traditional ML-based insights
            - charts: Generated charts (base64)
            - llm_interpretation: Natural language interpretation
        """

        query_lower = query.lower()

        # Determine analysis type based on query
        if any(word in query_lower for word in ['price', 'cost', 'expensive', 'cheap']):
            return self._price_analysis_with_interpretation(query)

        elif any(word in query_lower for word in ['rating', 'review', 'quality']):
            return self._rating_analysis_with_interpretation(query)

        elif any(word in query_lower for word in ['discount', 'offer', 'deal']):
            return self._discount_analysis_with_interpretation(query)

        elif any(word in query_lower for word in ['category', 'categories', 'segment']):
            return self._category_analysis_with_interpretation(query)

        elif any(word in query_lower for word in ['cluster', 'group', 'segment']):
            return self._clustering_analysis_with_interpretation(query)

        elif any(word in query_lower for word in ['trend', 'insight', 'analysis', 'overview']):
            return self._comprehensive_analysis_with_interpretation(query)

        else:
            # Default: comprehensive analysis
            return self._comprehensive_analysis_with_interpretation(query)

    def _price_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Price analysis with charts and LLM interpretation"""
        # ML Analysis
        ml_insights = self.analyzer.price_analysis()

        # Generate Charts
        chart_base64, chart_desc = self.chart_generator.generate_price_distribution_chart()

        # LLM Interpretation
        interpretation_prompt = f"""You are a data analyst providing insights from e-commerce sales data.

User Query: "{query}"

Data Analysis Results:
{chart_desc}

Detailed Statistics:
{ml_insights}

Provide a comprehensive, natural language interpretation that:
1. Directly answers the user's query
2. Highlights key insights from the data
3. Provides actionable recommendations
4. Uses business-friendly language

Keep your response concise (3-4 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Price Analysis',
            'ml_insights': ml_insights,
            'charts': [{'image': chart_base64, 'description': chart_desc}],
            'llm_interpretation': llm_response.text
        }

    def _rating_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Rating analysis with charts and LLM interpretation"""
        # Generate Charts
        chart_base64, chart_desc = self.chart_generator.generate_rating_analysis_chart()

        # ML insights
        behavior = self.analyzer.customer_behavior_analysis()

        interpretation_prompt = f"""You are a data analyst providing insights from e-commerce sales data.

User Query: "{query}"

Rating Analysis:
{chart_desc}

Customer Behavior Insights:
{behavior}

Provide a comprehensive interpretation focusing on:
1. Rating patterns and what they reveal
2. Customer preferences and behavior
3. Product quality indicators
4. Recommendations for improvement

Keep your response concise (3-4 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Rating Analysis',
            'ml_insights': behavior,
            'charts': [{'image': chart_base64, 'description': chart_desc}],
            'llm_interpretation': llm_response.text
        }

    def _discount_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Discount analysis with visualization"""
        chart_base64, chart_desc = self.chart_generator.generate_discount_analysis_chart()

        ml_insights = self.analyzer.price_analysis()

        interpretation_prompt = f"""You are a data analyst providing insights from e-commerce sales data.

User Query: "{query}"

Discount Analysis:
{chart_desc}

Price Statistics:
{ml_insights['discount_stats']}

Provide insights on:
1. Discount patterns and strategies
2. Price positioning
3. Recommendations for pricing optimization

Keep your response concise (2-3 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Discount Analysis',
            'ml_insights': ml_insights['discount_stats'],
            'charts': [{'image': chart_base64, 'description': chart_desc}],
            'llm_interpretation': llm_response.text
        }

    def _category_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Category-based analysis"""
        chart_base64, chart_desc = self.chart_generator.generate_sales_by_category_chart()

        trends = self.analyzer.sales_trends_analysis()

        interpretation_prompt = f"""You are a data analyst providing insights from e-commerce sales data.

User Query: "{query}"

Category Analysis:
{chart_desc}

Sales Trends:
Top categories and their performance

Provide insights on:
1. Category performance and trends
2. Market opportunities
3. Product mix recommendations

Keep your response concise (3 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Category Analysis',
            'ml_insights': trends.get('by_category', {}),
            'charts': [{'image': chart_base64, 'description': chart_desc}],
            'llm_interpretation': llm_response.text
        }

    def _clustering_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Clustering analysis with ML and visualization"""
        # Perform clustering
        clustering_results = self.analyzer.perform_clustering(n_clusters=4)

        # Create temporary dataframe with cluster info
        clustered_df = pd.DataFrame(clustering_results['clustered_data'])
        chart_base64, chart_desc = self.chart_generator.generate_clustering_chart(
            self.df.assign(cluster=0)  # Placeholder
        )

        interpretation_prompt = f"""You are a data analyst providing insights from e-commerce sales data.

User Query: "{query}"

Clustering Analysis:
{clustering_results}

Provide insights on:
1. Product segments identified
2. Characteristics of each cluster
3. Business strategies for each segment

Keep your response concise (3-4 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Clustering Analysis',
            'ml_insights': clustering_results,
            'charts': [{'image': chart_base64, 'description': chart_desc}],
            'llm_interpretation': llm_response.text
        }

    def _comprehensive_analysis_with_interpretation(self, query: str) -> Dict[str, Any]:
        """Comprehensive analysis with multiple charts"""
        charts = []

        # Generate multiple charts
        price_chart, price_desc = self.chart_generator.generate_price_distribution_chart()
        charts.append({'image': price_chart, 'description': price_desc})

        category_chart, cat_desc = self.chart_generator.generate_sales_by_category_chart()
        if category_chart:
            charts.append({'image': category_chart, 'description': cat_desc})

        rating_chart, rating_desc = self.chart_generator.generate_rating_analysis_chart()
        charts.append({'image': rating_chart, 'description': rating_desc})

        # Comprehensive ML insights
        ml_insights = {
            'price_analysis': self.analyzer.price_analysis(),
            'trends': self.analyzer.sales_trends_analysis(),
            'behavior': self.analyzer.customer_behavior_analysis()
        }

        interpretation_prompt = f"""You are a data analyst providing comprehensive insights from e-commerce sales data.

User Query: "{query}"

Comprehensive Analysis Summary:
1. {price_desc}
2. {cat_desc if category_chart else 'Category data not available'}
3. {rating_desc}

Full ML Insights:
{ml_insights}

Provide a comprehensive business intelligence report that:
1. Answers the user's query with data-driven insights
2. Identifies key trends and patterns
3. Provides strategic recommendations
4. Highlights opportunities and areas for improvement

Keep your response well-structured (4-5 paragraphs)."""

        llm_response = self.gemini_model.generate_content(interpretation_prompt)

        return {
            'analysis_type': 'Comprehensive Analysis',
            'ml_insights': ml_insights,
            'charts': charts,
            'llm_interpretation': llm_response.text
        }
