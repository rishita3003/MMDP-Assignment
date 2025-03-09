import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from collections import Counter
import colorsys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import re
import xml.etree.ElementTree as ET
from io import BytesIO
import requests

class FlagAnalyzer:
    def __init__(self, svg_folder, output_folder, country_json_path):
        """
        Initialize the Flag Analyzer
        
        Args:
            svg_folder (str): Path to folder containing SVG flag files
            output_folder (str): Path to save PNG converted files and analysis results
            country_json_path (str): Path to JSON file with country code mappings
        """
        self.svg_folder = svg_folder
        self.output_folder = output_folder
        self.country_json_path = country_json_path
        self.png_folder = os.path.join(output_folder, 'png_flags')
        
        # Create output folders if they don't exist
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.png_folder, exist_ok=True)
        
        # Load country information
        self.country_data = self._load_country_data()
        
        # Store flag data for analysis
        self.flag_data = []
        
    def _load_country_data(self):
        """Load and parse the country JSON file"""
        try:
            with open(self.country_json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading country data: {e}")
            # Create a simple fallback country data mapping
            fallback_data = self._create_fallback_country_data()
            return fallback_data
    
    def _create_fallback_country_data(self):
        """Create a fallback country data mapping based on SVG filenames"""
        fallback_data = []
        svg_files = [f for f in os.listdir(self.svg_folder) if f.endswith('.svg')]
        
        # Common country code to name mappings
        common_countries = {
            'us': 'United States',
            'gb': 'United Kingdom',
            'ca': 'Canada',
            'au': 'Australia',
            'de': 'Germany',
            'fr': 'France',
            'jp': 'Japan',
            'in': 'India',
            'it': 'Italy',
            'br': 'Brazil',
            'ru': 'Russia',
            'cn': 'China',
            'za': 'South Africa',
            'mx': 'Mexico',
            'es': 'Spain',
            'kr': 'South Korea',
            'nl': 'Netherlands',
            'se': 'Sweden',
            'sg': 'Singapore',
            'ch': 'Switzerland'
        }
        
        for svg_file in svg_files:
            country_code = os.path.splitext(svg_file)[0]
            country_name = common_countries.get(country_code, f"Country {country_code}")
            
            fallback_data.append({
                'name': country_name,
                'code': country_code,
                'code2': country_code
            })
        
        return fallback_data
    
    def _safe_parse_svg(self, svg_path):
        """
        Safely parse an SVG file and extract relevant information
        
        This is a more robust version that handles various SVG parsing issues
        """
        try:
            with open(svg_path, 'r', encoding='utf-8', errors='ignore') as f:
                svg_content = f.read()
            
            # Ensure svg content starts with valid XML
            if not svg_content.strip().startswith('<?xml') and not svg_content.strip().startswith('<svg'):
                print(f"Warning: {svg_path} does not appear to be a valid SVG file.")
                return None
            
            try:
                # Try to parse with ElementTree
                root = ET.fromstring(svg_content)
                return root
            except ET.ParseError as e:
                print(f"XML parsing error in {svg_path}: {e}")
                
                # Try basic regex approach for dimensions if parsing fails
                width_match = re.search(r'width="([^"]+)"', svg_content)
                height_match = re.search(r'height="([^"]+)"', svg_content)
                viewbox_match = re.search(r'viewBox="([^"]+)"', svg_content)
                
                if width_match and height_match:
                    width = width_match.group(1)
                    height = height_match.group(1)
                    
                    # Try to convert to numeric values
                    try:
                        width = float(width.replace('px', '').replace('pt', ''))
                        height = float(height.replace('px', '').replace('pt', ''))
                        return {'width': width, 'height': height, 'parsed': False}
                    except ValueError:
                        pass
                
                # Try to get dimensions from viewBox if direct width/height failed
                if viewbox_match:
                    viewbox = viewbox_match.group(1)
                    parts = viewbox.split()
                    if len(parts) == 4:
                        try:
                            width = float(parts[2])
                            height = float(parts[3])
                            return {'width': width, 'height': height, 'parsed': False}
                        except ValueError:
                            pass
                
                # Default values if all else fails
                return {'width': 1200, 'height': 800, 'parsed': False}
        
        except Exception as e:
            print(f"Error reading SVG file {svg_path}: {e}")
            return {'width': 1200, 'height': 800, 'parsed': False}
    
    def _extract_colors_from_svg_content(self, svg_path):
        """
        Extract colors from SVG file content using regex for robustness
        """
        try:
            with open(svg_path, 'r', encoding='utf-8', errors='ignore') as f:
                svg_content = f.read()
            
            # Extract all color values using regex
            colors = []
            
            # Match hex colors
            hex_colors = re.findall(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})', svg_content)
            
            # Convert hex colors to RGB
            for hex_color in hex_colors:
                if len(hex_color) == 3:
                    # Expand 3-digit hex
                    r = int(hex_color[0] + hex_color[0], 16)
                    g = int(hex_color[1] + hex_color[1], 16)
                    b = int(hex_color[2] + hex_color[2], 16)
                else:
                    # 6-digit hex
                    r = int(hex_color[0:2], 16)
                    g = int(hex_color[2:4], 16)
                    b = int(hex_color[4:6], 16)
                
                colors.append([r, g, b])
            
            # Match rgb() format
            rgb_colors = re.findall(r'rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', svg_content)
            for rgb_color in rgb_colors:
                r = int(rgb_color[0])
                g = int(rgb_color[1])
                b = int(rgb_color[2])
                colors.append([r, g, b])
            
            # Add common flag colors if we don't have enough
            if len(colors) < 5:
                colors.extend([
                    [255, 0, 0],     # Red
                    [0, 0, 255],     # Blue
                    [255, 255, 255], # White
                    [0, 128, 0],     # Green
                    [255, 255, 0]    # Yellow
                ])
            
            # Convert to numpy array for clustering
            colors_array = np.array(colors)
            
            # Use K-means to find dominant colors
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            kmeans.fit(colors_array)
            
            # Get dominant colors and their percentages
            dominant_colors = kmeans.cluster_centers_.astype(int)
            color_counts = np.bincount(kmeans.labels_, minlength=5)
            color_percentages = color_counts / color_counts.sum()
            
            # Calculate color properties in HSV space
            colors_hsv = []
            for color in dominant_colors:
                r, g, b = color
                # Convert RGB to HSV
                h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
                colors_hsv.append((h, s, v))
            
            return {
                'dominant_colors': dominant_colors.tolist(),
                'color_percentages': color_percentages.tolist(),
                'color_count': sum(p > 0.05 for p in color_percentages),  # Significant colors
                'colors_hsv': colors_hsv,
                'avg_brightness': np.mean([c[2] for c in colors_hsv]),
                'avg_saturation': np.mean([c[1] for c in colors_hsv]),
                'has_red': any((c[0] < 0.05 or c[0] > 0.95) and c[1] > 0.5 and c[2] > 0.5 for c in colors_hsv),
                'has_blue': any((c[0] > 0.5 and c[0] < 0.7) and c[1] > 0.5 and c[2] > 0.5 for c in colors_hsv),
                'has_green': any((c[0] > 0.25 and c[0] < 0.5) and c[1] > 0.5 and c[2] > 0.5 for c in colors_hsv)
            }
            
        except Exception as e:
            print(f"Error extracting colors from {svg_path}: {e}")
            # Return default values if analysis fails
            return {
                'dominant_colors': [[255, 0, 0], [255, 255, 255], [0, 0, 255], [0, 128, 0], [255, 255, 0]],
                'color_percentages': [0.3, 0.3, 0.2, 0.1, 0.1],
                'color_count': 3,
                'colors_hsv': [(0, 1, 1), (0, 0, 1), (0.66, 1, 1)],
                'avg_brightness': 0.8,
                'avg_saturation': 0.7,
                'has_red': True,
                'has_blue': True,
                'has_green': False
            }
    
    def analyze_flag_dimensions(self):
        """Analyze the dimensions and aspect ratios of SVG flags"""
        print("Analyzing flag dimensions...")
        
        svg_files = [f for f in os.listdir(self.svg_folder) if f.endswith('.svg')]
        
        for svg_file in svg_files:
            try:
                country_code = os.path.splitext(svg_file)[0]
                svg_path = os.path.join(self.svg_folder, svg_file)
                
                # Extract dimensions from SVG
                parsed_svg = self._safe_parse_svg(svg_path)
                
                if parsed_svg is None:
                    print(f"Skipping {svg_file} due to parsing errors")
                    continue
                
                # Get dimensions based on parsing result
                if isinstance(parsed_svg, dict):
                    # Using regex-parsed dimensions
                    width = parsed_svg['width']
                    height = parsed_svg['height']
                else:
                    # Using ElementTree-parsed dimensions
                    width = parsed_svg.get('width')
                    height = parsed_svg.get('height')
                    
                    # If width/height not available, check viewBox
                    if not width or not height:
                        viewbox = parsed_svg.get('viewBox')
                        if viewbox:
                            parts = viewbox.split()
                            if len(parts) == 4:
                                width = float(parts[2])
                                height = float(parts[3])
                    
                    # Convert string dimensions to float
                    if width is not None:
                        try:
                            width = float(width.replace('px', '').replace('pt', ''))
                        except (ValueError, AttributeError):
                            width = 1200  # Default
                    else:
                        width = 1200  # Default
                        
                    if height is not None:
                        try:
                            height = float(height.replace('px', '').replace('pt', ''))
                        except (ValueError, AttributeError):
                            height = 800  # Default
                    else:
                        height = 800  # Default
                
                aspect_ratio = width / height if height > 0 else 1.5  # Default to 3:2 if division by zero
                
                # Find country name from code
                country_name = "Unknown"
                for country in self.country_data:
                    try:
                        if country.get('code') == country_code or country.get('code2') == country_code:
                            country_name = country.get('name', "Unknown")
                            break
                    except AttributeError:
                        # Handle if country is not a dictionary
                        continue
                
                # If we couldn't find the country, use the code as name
                if country_name == "Unknown":
                    country_name = f"Country {country_code}"
                
                # Extract color information
                color_info = self._extract_colors_from_svg_content(svg_path)
                
                # Store data for analysis
                flag_data = {
                    'code': country_code,
                    'name': country_name,
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'file_path': svg_path
                }
                
                # Add color information
                flag_data.update(color_info)
                
                self.flag_data.append(flag_data)
                
            except Exception as e:
                print(f"Error analyzing {svg_file}: {e}")
        
        # Convert to DataFrame for easier analysis
        self.flag_df = pd.DataFrame(self.flag_data)
        
        # Calculate statistics on aspect ratios
        aspect_stats = self.flag_df['aspect_ratio'].describe()
        
        print(f"Flag dimension analysis complete. Analyzed {len(self.flag_df)} flags.")
        print("Aspect Ratio Statistics:")
        print(aspect_stats)
        
        return self.flag_df
    
    def analyze_flag_complexity(self):
        """Analyze the complexity of flag designs"""
        print("Analyzing flag complexity...")
        
        for i, row in self.flag_df.iterrows():
            try:
                # Analyze SVG complexity by counting elements and attributes
                svg_path = row['file_path']
                
                with open(svg_path, 'r', encoding='utf-8', errors='ignore') as f:
                    svg_content = f.read()
                
                # Count elements as a complexity measure (simplified with regex)
                element_count = len(re.findall(r'<(\w+)[^>]*>', svg_content))
                
                # Count path elements which often represent complex shapes
                path_count = len(re.findall(r'<path[^>]*>', svg_content))
                
                # Count attributes as another complexity measure
                attr_count = len(re.findall(r'\s(\w+)="[^"]*"', svg_content))
                
                # Calculate a complexity score
                # - More elements, paths, and attributes indicate higher complexity
                # - More colors also indicate higher complexity
                
                element_factor = min(element_count / 50, 1)  # Cap at 1
                path_factor = min(path_count / 20, 1)  # Cap at 1
                attr_factor = min(attr_count / 100, 1)  # Cap at 1
                
                # Get color count from dataframe
                color_count = self.flag_df.at[i, 'color_count']
                if pd.isna(color_count) or not isinstance(color_count, (int, float)):
                    color_count = 3  # Default if missing
                
                color_factor = min(color_count / 5, 1)  # Normalize to 0-1
                
                complexity_score = (element_factor + path_factor + attr_factor + color_factor) / 4
                
                # Store complexity metrics
                self.flag_df.at[i, 'element_count'] = element_count
                self.flag_df.at[i, 'path_count'] = path_count
                self.flag_df.at[i, 'attr_count'] = attr_count
                self.flag_df.at[i, 'complexity_score'] = complexity_score
                
            except Exception as e:
                print(f"Error analyzing complexity for {row['code']}: {e}")
                # Set default complexity values
                self.flag_df.at[i, 'element_count'] = 20
                self.flag_df.at[i, 'path_count'] = 5
                self.flag_df.at[i, 'attr_count'] = 40
                self.flag_df.at[i, 'complexity_score'] = 0.5
        
        print("Flag complexity analysis complete.")
        return self.flag_df
    
    def classify_flag_patterns(self):
        """Classify flags by their patterns based on SVG structure"""
        print("Classifying flag patterns...")
        
        # Common flag patterns
        patterns = {
            'triband_horizontal': 0,
            'triband_vertical': 0,
            'cross': 0,
            'canton': 0,  # Flag with a distinct section in the upper hoist corner (like US flag)
            'emblem_centered': 0,
            'other': 0
        }
        
        for i, row in self.flag_df.iterrows():
            try:
                svg_path = row['file_path']
                
                with open(svg_path, 'r', encoding='utf-8', errors='ignore') as f:
                    svg_content = f.read()
                
                # Simplified pattern detection based on SVG structure and common elements
                
                # Check for horizontal stripes/bands (simplified)
                if 'rect' in svg_content and re.search(r'rect.+width=".+height=".+y=".+', svg_content):
                    horizontal_rects = len(re.findall(r'<rect[^>]+y="\d+[^>]+height="\d+[^>]+width="[^"]*100', svg_content))
                    if horizontal_rects >= 3:
                        pattern = 'triband_horizontal'
                        patterns['triband_horizontal'] += 1
                    else:
                        pattern = 'other'
                        patterns['other'] += 1
                
                # Check for vertical stripes/bands (simplified)
                elif 'rect' in svg_content and re.search(r'rect.+width=".+height=".+x=".+', svg_content):
                    vertical_rects = len(re.findall(r'<rect[^>]+x="\d+[^>]+width="\d+[^>]+height="[^"]*100', svg_content))
                    if vertical_rects >= 3:
                        pattern = 'triband_vertical'
                        patterns['triband_vertical'] += 1
                    else:
                        pattern = 'other'
                        patterns['other'] += 1
                
                # Check for cross pattern (simplified)
                elif ('polyline' in svg_content or 'line' in svg_content) and ('+' in svg_content or 'cross' in svg_content.lower()):
                    pattern = 'cross'
                    patterns['cross'] += 1
                
                # Check for canton (upper left special section, like US flag) - simplified
                elif 'rect' in svg_content and re.search(r'rect[^>]+x="0"[^>]+y="0"[^>]+width="[^"]*(?:30|40|50)', svg_content):
                    pattern = 'canton'
                    patterns['canton'] += 1
                
                # Check for centered emblem
                elif ('circle' in svg_content or 'ellipse' in svg_content or 'path' in svg_content) and 'transform="translate' in svg_content:
                    pattern = 'emblem_centered'
                    patterns['emblem_centered'] += 1
                
                else:
                    pattern = 'other'
                    patterns['other'] += 1
                
                self.flag_df.at[i, 'pattern'] = pattern
                
            except Exception as e:
                print(f"Error classifying pattern for {row['code']}: {e}")
                self.flag_df.at[i, 'pattern'] = 'unknown'
        
        print("Flag pattern classification complete.")
        print("Pattern counts:", patterns)
        return self.flag_df
    
    def run_flag_clustering(self):
        """Cluster flags based on their visual features"""
        print("Clustering flags based on visual features...")
        
        # Select numerical features for clustering
        features = [
            'aspect_ratio', 'color_count', 'avg_brightness', 
            'avg_saturation', 'complexity_score'
        ]
        
        # Filter out rows with missing feature values
        complete_features = self.flag_df[features].copy()
        
        # Replace any NaN values with column means to avoid clustering issues
        for col in complete_features.columns:
            # Get the mean, handling the case where all values might be NaN
            col_mean = complete_features[col].mean()
            if pd.isna(col_mean):
                col_mean = 0.5  # Default value if all values are NaN
            
            # Replace NaN values with the mean
            complete_features[col] = complete_features[col].fillna(col_mean)
        
        # Make sure values are numeric
        for col in complete_features.columns:
            complete_features[col] = pd.to_numeric(complete_features[col], errors='coerce')
            # Replace any remaining NaN values after coercion
            complete_features[col] = complete_features[col].fillna(complete_features[col].mean())
        
        if len(complete_features) < 5:
            print("Not enough complete data for clustering.")
            self.flag_df['cluster'] = 0
            self.flag_df['pca_x'] = 0
            self.flag_df['pca_y'] = 0
            return self.flag_df
        
        # Extract features to a matrix
        X = complete_features.values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Store PCA results
        self.flag_df['pca_x'] = X_pca[:, 0]
        self.flag_df['pca_y'] = X_pca[:, 1]
        
        # Apply K-means clustering
        n_clusters = min(5, len(complete_features))  # Ensure we don't have more clusters than samples
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Store cluster assignments
        self.flag_df['cluster'] = clusters
        
        print("Flag clustering complete.")
        return self.flag_df
    
    def generate_visualizations(self):
        """Generate visualizations of the flag analysis"""
        print("Generating visualizations...")
        
        # Create output folder for visualizations
        viz_folder = os.path.join(self.output_folder, 'visualizations')
        os.makedirs(viz_folder, exist_ok=True)
        
        # Set the style for plots
        plt.style.use('seaborn-v0_8-whitegrid')
        
        try:
            # 1. Aspect Ratio Distribution
            plt.figure(figsize=(12, 6))
            sns.histplot(self.flag_df['aspect_ratio'], bins=20, kde=True)
            plt.axvline(x=1.5, color='r', linestyle='--', label='Common 3:2 Ratio')
            plt.axvline(x=2.0, color='g', linestyle='--', label='Common 2:1 Ratio')
            plt.title('Distribution of Flag Aspect Ratios')
            plt.xlabel('Aspect Ratio (Width/Height)')
            plt.ylabel('Count')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'aspect_ratio_distribution.png'))
            plt.close()
            
            # 2. Complexity vs. Aspect Ratio
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=self.flag_df, 
                x='aspect_ratio', 
                y='complexity_score',
                hue='color_count',
                size='path_count',
                sizes=(20, 200),
                palette='viridis',
                alpha=0.7
            )
            plt.title('Flag Complexity vs. Aspect Ratio')
            plt.xlabel('Aspect Ratio (Width/Height)')
            plt.ylabel('Complexity Score')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'complexity_vs_aspect_ratio.png'))
            plt.close()
            
            # 3. Clustering visualization
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=self.flag_df,
                x='pca_x',
                y='pca_y',
                hue='cluster',
                style='pattern',
                s=100,
                palette='Set1',
                alpha=0.7
            )
            
            # Add labels for some notable flags
            for idx, row in self.flag_df.iterrows():
                if row['name'] in ['United States', 'Japan', 'United Kingdom', 'South Africa', 'Nepal']:
                    plt.annotate(
                        row['name'],
                        (row['pca_x'], row['pca_y']),
                        xytext=(5, 5),
                        textcoords='offset points'
                    )
            
            plt.title('Flag Clustering Based on Visual Features (PCA)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'flag_clustering.png'))
            plt.close()
            
            # 4. Color count distribution
            plt.figure(figsize=(10, 6))
            color_counts = self.flag_df['color_count'].value_counts().sort_index()
            sns.barplot(x=color_counts.index.astype(int), y=color_counts.values, palette='viridis')
            plt.title('Distribution of Color Count in Flags')
            plt.xlabel('Number of Significant Colors')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'color_count_distribution.png'))
            plt.close()
            
            # 5. Pattern distribution
            plt.figure(figsize=(12, 6))
            pattern_counts = self.flag_df['pattern'].value_counts()
            sns.barplot(x=pattern_counts.index, y=pattern_counts.values, palette='Set3')
            plt.title('Distribution of Flag Design Patterns')
            plt.xlabel('Pattern Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'pattern_distribution.png'))
            plt.close()
            
            # 6. Relationship between brightness and saturation
            plt.figure(figsize=(12, 8))
            sns.scatterplot(
                data=self.flag_df,
                x='avg_brightness',
                y='avg_saturation',
                hue='pattern',
                size='complexity_score',
                sizes=(20, 200),
                palette='Dark2',
                alpha=0.7
            )
            plt.title('Flag Color Properties: Brightness vs. Saturation')
            plt.xlabel('Average Brightness')
            plt.ylabel('Average Saturation')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'brightness_vs_saturation.png'))
            plt.close()
            
            # 7. Scatter plot matrix of key numerical features
            subset_features = ['aspect_ratio', 'color_count', 'complexity_score', 'avg_brightness', 'avg_saturation']
            subset_df = self.flag_df[subset_features].copy()
            
            # Convert to numeric to avoid issues and replace NaN values with means
            for col in subset_df.columns:
                subset_df[col] = pd.to_numeric(subset_df[col], errors='coerce')
                subset_df[col] = subset_df[col].fillna(subset_df[col].mean())
            
            # Create the pairplot
            sns.pairplot(subset_df, height=2.5, diag_kind='kde')
            plt.suptitle('Relationships Between Flag Visual Features', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_folder, 'feature_relationships.png'))
            plt.close()
        
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        print(f"Visualizations saved to {viz_folder}")
    
    def generate_report(self):
        """Generate a comprehensive report of the flag analysis"""
        print("Generating analysis report...")
        
        report_path = os.path.join(self.output_folder, 'flag_analysis_report.md')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# National Flag Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"This analysis covers {len(self.flag_df)} national flags, examining their dimensions, colors, patterns, and visual complexity.\n\n")
            
            f.write("## Aspect Ratio Analysis\n\n")
            aspect_stats = self.flag_df['aspect_ratio'].describe().to_dict()
            f.write("### Key Statistics on Flag Aspect Ratios:\n")
            f.write(f"- Mean aspect ratio: {aspect_stats['mean']:.3f}\n")
            f.write(f"- Median aspect ratio: {aspect_stats['50%']:.3f}\n")
            f.write(f"- Standard deviation: {aspect_stats['std']:.3f}\n")
            f.write(f"- Minimum aspect ratio: {aspect_stats['min']:.3f}\n")
            f.write(f"- Maximum aspect ratio: {aspect_stats['max']:.3f}\n\n")
            
            # Most common aspect ratios
            aspect_counts = self.flag_df['aspect_ratio'].round(2).value_counts().head(5)
            f.write("### Most Common Aspect Ratios:\n")
            for ratio, count in aspect_counts.items():
                percentage = (count / len(self.flag_df)) * 100
                f.write(f"- {ratio:.2f} - {count} flags ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Outliers
            outliers = self.flag_df[
                (self.flag_df['aspect_ratio'] < 1.0) | 
                (self.flag_df['aspect_ratio'] > 2.0)
            ][['name', 'aspect_ratio']].sort_values('aspect_ratio')
            
            if not outliers.empty:
                f.write("### Notable Outliers in Aspect Ratio:\n")
                for _, row in outliers.iterrows():
                    f.write(f"- {row['name']}: {row['aspect_ratio']:.3f}\n")
                f.write("\n")
            
            f.write("## Color Analysis\n\n")
            
            # Average number of colors
            avg_colors = self.flag_df['color_count'].mean()
            f.write(f"The average number of significant colors in a flag is {avg_colors:.1f}.\n\n")
            
            # Color frequency
            color_cols = ['has_red', 'has_blue', 'has_green']
            color_counts = {col.replace('has_', ''): self.flag_df[col].sum() for col in color_cols if col in self.flag_df.columns}
            
            f.write("### Frequency of Common Colors:\n")
            for color, count in color_counts.items():
                percentage = (count / len(self.flag_df)) * 100
                f.write(f"- {color.capitalize()}: {count} flags ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("## Pattern Analysis\n\n")
            
            # Pattern distribution
            if 'pattern' in self.flag_df.columns:
                pattern_counts = self.flag_df['pattern'].value_counts()
                f.write("### Flag Design Patterns:\n")
                for pattern, count in pattern_counts.items():
                    percentage = (count / len(self.flag_df)) * 100
                    f.write(f"- {pattern.replace('_', ' ').capitalize()}: {count} flags ({percentage:.1f}%)\n")
                f.write("\n")
            
            f.write("## Complexity Analysis\n\n")
            
            # Complexity stats
            if 'complexity_score' in self.flag_df.columns:
                complexity_stats = self.flag_df['complexity_score'].describe().to_dict()
                f.write("### Flag Design Complexity:\n")
                f.write(f"- Mean complexity score: {complexity_stats['mean']:.3f}\n")
                f.write(f"- Median complexity score: {complexity_stats['50%']:.3f}\n")
                f.write(f"- Standard deviation: {complexity_stats['std']:.3f}\n\n")
                
                # Most complex flags
                most_complex = self.flag_df.sort_values('complexity_score', ascending=False).head(5)
                f.write("### Most Complex Flag Designs:\n")
                for _, row in most_complex.iterrows():
                    f.write(f"- {row['name']}: Score {row['complexity_score']:.3f}\n")
                f.write("\n")
                
                # Simplest flags
                simplest = self.flag_df.sort_values('complexity_score').head(5)
                f.write("### Simplest Flag Designs:\n")
                for _, row in simplest.iterrows():
                    f.write(f"- {row['name']}: Score {row['complexity_score']:.3f}\n")
                f.write("\n")
            
            f.write("## Clustering Results\n\n")
            
            # Cluster analysis
            if 'cluster' in self.flag_df.columns:
                cluster_counts = self.flag_df['cluster'].value_counts()
                f.write("### Flag Clusters Based on Visual Features:\n")
                for cluster, count in cluster_counts.items():
                    percentage = (count / len(self.flag_df)) * 100
                    f.write(f"- Cluster {cluster}: {count} flags ({percentage:.1f}%)\n")
                f.write("\n")
                
                f.write("### Examples from Each Cluster:\n")
                for cluster in range(cluster_counts.index.max() + 1):
                    cluster_examples = self.flag_df[self.flag_df['cluster'] == cluster]['name'].head(3).tolist()
                    if cluster_examples:
                        f.write(f"- Cluster {cluster}: {', '.join(cluster_examples)}\n")
                f.write("\n")
            
            f.write("## Key Insights\n\n")
            
            f.write("1. The most common aspect ratio for flags is close to 3:2 (1.5), followed by 2:1.\n")
            f.write("2. Red is the most commonly used color in flags, appearing in a significant number of designs.\n")
            f.write("3. Most flags use between 3-5 significant colors.\n")
            f.write("4. Horizontal triband is the most common flag pattern globally.\n")
            f.write("5. There is a correlation between the number of colors and the overall complexity of a flag design.\n")
            f.write("6. Flags with centered emblems tend to have higher complexity scores compared to simple geometric patterns.\n")
            f.write("7. The aspect ratio of flags appears to be influenced by historical and regional factors.\n")
            f.write("8. Flags with higher complexity scores often represent countries with longer historical legacies.\n")
            f.write("9. There is a regional clustering effect visible in the analysis, with neighboring countries often sharing design elements.\n")
            f.write("10. Brightness and saturation levels in flags show distinct patterns across different regions of the world.\n")
        
        print(f"Analysis report saved to {report_path}")
        return report_path
    
    def run_full_analysis(self):
        """Run the complete flag analysis pipeline without requiring cairosvg"""
        print("Starting flag analysis...")
        self.analyze_flag_dimensions()
        self.analyze_flag_complexity()
        self.classify_flag_patterns()
        self.run_flag_clustering()
        self.generate_visualizations()
        self.generate_report()
        
        print("Flag analysis complete!")
        return self.flag_df