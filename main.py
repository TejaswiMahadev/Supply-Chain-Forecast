import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib

# Machine Learning Libraries
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

# Time Series Libraries
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

# Causal Inference
try:
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print("Warning: DoWhy not available. Install with: pip install dowhy")

# For causal graphs (alternative to pgmpy)
import networkx as nx

# Streamlit for dashboard
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#interpretation 
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY  not found in .env file")

genai.configure(api_key=api_key)



class SupplyChainDataGenerator:
    """Generate realistic supply chain data for testing"""
    
    def __init__(self, start_date='2022-01-01', end_date='2024-12-31', n_products=10):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_products = n_products
        self.products = [f'P{str(i).zfill(3)}' for i in range(1, n_products + 1)]
        
    def generate_data(self) -> pd.DataFrame:
        """Generate comprehensive supply chain dataset"""
        np.random.seed(42)
        
        # Create date range
        date_range = pd.date_range(self.start_date, self.end_date, freq='W')
        
        data = []
        
        for product in self.products:
            # Product-specific parameters
            base_demand = np.random.randint(100, 500)
            seasonality_strength = np.random.uniform(0.1, 0.3)
            trend_slope = np.random.uniform(-0.5, 1.0)
            supplier_reliability = np.random.uniform(0.7, 0.95)
            
            # Initialize stock level
            current_stock = np.random.randint(200, 1000)
            
            for i, date in enumerate(date_range):
                # Time-based features
                week_of_year = date.week
                month = date.month
                quarter = (month - 1) // 3 + 1
                
                # Seasonality effect
                seasonal_factor = 1 + seasonality_strength * np.sin(2 * np.pi * week_of_year / 52)
                
                # Trend effect
                trend_factor = 1 + trend_slope * i / len(date_range)
                
                # Demand calculation
                base_demand_adj = base_demand * seasonal_factor * trend_factor
                demand_noise = np.random.normal(0, base_demand_adj * 0.1)
                
                # Promotional effect (random promotions)
                promo_active = np.random.choice([0, 1], p=[0.8, 0.2])
                promo_multiplier = 1.5 if promo_active else 1.0
                
                demand = max(0, base_demand_adj * promo_multiplier + demand_noise)
                
                # Supplier delay (correlated with external factors)
                supplier_delay_prob = 1 - supplier_reliability
                if i > 0 and data[i-1]['Supplier_Delay'] == 1:
                    supplier_delay_prob *= 1.3  # Delays tend to cluster
                
                supplier_delay = np.random.choice([0, 1], p=[1-supplier_delay_prob, supplier_delay_prob])
                
                # Order quantity (responsive to stock levels and demand)
                if current_stock < base_demand * 0.5:  # Low stock triggers larger orders
                    order_quantity = base_demand * np.random.uniform(1.5, 2.5)
                elif current_stock > base_demand * 2:  # High stock reduces orders
                    order_quantity = base_demand * np.random.uniform(0.3, 0.7)
                else:
                    order_quantity = base_demand * np.random.uniform(0.8, 1.2)
                
                # Lead time effect on delivery
                lead_time = 1 if supplier_delay == 0 else np.random.randint(2, 4)
                
                # Stock level calculation
                if i >= lead_time:
                    delivered_quantity = data[i-lead_time]['Order_Quantity'] if not supplier_delay else data[i-lead_time]['Order_Quantity'] * 0.7
                else:
                    delivered_quantity = 0
                
                current_stock = max(0, current_stock + delivered_quantity - demand)
                
                # Stockout determination
                stockout = 1 if current_stock <= 0 else 0
                
                # Additional features
                days_since_last_order = i % 7 + 1  # Weekly ordering pattern
                competitor_promo = np.random.choice([0, 1], p=[0.9, 0.1])
                weather_impact = np.random.choice([0, 1], p=[0.95, 0.05])  # Rare weather events
                
                data.append({
                    'Date': date,
                    'Product_ID': product,
                    'Stock_Level': current_stock,
                    'Supplier_Delay': supplier_delay,
                    'Promo_Active': promo_active,
                    'Order_Quantity': order_quantity,
                    'Demand': demand,
                    'Stockout': stockout,
                    'Lead_Time': lead_time,
                    'Week_of_Year': week_of_year,
                    'Month': month,
                    'Quarter': quarter,
                    'Days_Since_Last_Order': days_since_last_order,
                    'Competitor_Promo': competitor_promo,
                    'Weather_Impact': weather_impact,
                    'Seasonal_Factor': seasonal_factor,
                    'Trend_Factor': trend_factor
                })
        
        return pd.DataFrame(data)

class FeatureEngineer:
    """Advanced feature engineering for supply chain data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def create_lag_features(self, df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
        """Create lag features for specified columns"""
        df_features = df.copy()
        
        for product in df['Product_ID'].unique():
            product_data = df[df['Product_ID'] == product].sort_values('Date')
            
            for col in columns:
                for lag in lags:
                    lag_col_name = f'{col}_lag_{lag}'
                    df_features.loc[df_features['Product_ID'] == product, lag_col_name] = \
                        product_data[col].shift(lag).values
        
        return df_features
    
    def create_rolling_features(self, df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
        """Create rolling statistics features"""
        df_features = df.copy()
        
        for product in df['Product_ID'].unique():
            product_mask = df_features['Product_ID'] == product
            product_data = df_features[product_mask].sort_values('Date')
            
            for col in columns:
                for window in windows:
                    # Rolling mean
                    roll_mean_col = f'{col}_rolling_mean_{window}'
                    df_features.loc[product_mask, roll_mean_col] = \
                        product_data[col].rolling(window=window, min_periods=1).mean().values
                    
                    # Rolling std
                    roll_std_col = f'{col}_rolling_std_{window}'
                    df_features.loc[product_mask, roll_std_col] = \
                        product_data[col].rolling(window=window, min_periods=1).std().fillna(0).values
                    
                    # Rolling max
                    roll_max_col = f'{col}_rolling_max_{window}'
                    df_features.loc[product_mask, roll_max_col] = \
                        product_data[col].rolling(window=window, min_periods=1).max().values
                    
                    # Rolling min
                    roll_min_col = f'{col}_rolling_min_{window}'
                    df_features.loc[product_mask, roll_min_col] = \
                        product_data[col].rolling(window=window, min_periods=1).min().values
        
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        df_features = df.copy()
        
        # Demand-Supply interaction
        df_features['Demand_Stock_Ratio'] = df_features['Demand'] / (df_features['Stock_Level'] + 1)
        df_features['Order_Demand_Ratio'] = df_features['Order_Quantity'] / (df_features['Demand'] + 1)
        
        # Promotional interactions
        df_features['Promo_Demand_Interaction'] = df_features['Promo_Active'] * df_features['Demand']
        df_features['Promo_Stock_Interaction'] = df_features['Promo_Active'] * df_features['Stock_Level']
        
        # Supplier delay interactions
        df_features['Delay_Order_Interaction'] = df_features['Supplier_Delay'] * df_features['Order_Quantity']
        df_features['Delay_Stock_Interaction'] = df_features['Supplier_Delay'] * df_features['Stock_Level']
        
        # Time-based interactions
        df_features['Month_Demand_Interaction'] = df_features['Month'] * df_features['Demand']
        df_features['Quarter_Stock_Interaction'] = df_features['Quarter'] * df_features['Stock_Level']
        
        return df_features
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Starting feature engineering...")
        
        # Sort by product and date
        df = df.sort_values(['Product_ID', 'Date'])
        
        # Create lag features
        lag_columns = ['Stock_Level', 'Demand', 'Order_Quantity', 'Supplier_Delay']
        df = self.create_lag_features(df, lag_columns, [1, 2, 3, 4])
        
        # Create rolling features
        rolling_columns = ['Stock_Level', 'Demand', 'Order_Quantity']
        df = self.create_rolling_features(df, rolling_columns, [2, 4, 8])
        
        # Create interaction features
        df = self.create_interaction_features(df)
        
        # Encode categorical variables
        categorical_columns = ['Product_ID']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
        
        print(f"Feature engineering complete. Dataset shape: {df.shape}")
        return df

class CausalAnalyzer:
    """Causal analysis for supply chain relationships"""
    
    def __init__(self):
        self.causal_graph = nx.DiGraph()
        self.causal_model = None
        
    def build_causal_graph(self) -> nx.DiGraph:
        """Build causal graph based on domain knowledge"""
        # Define causal relationships
        edges = [
            ('Supplier_Delay', 'Stock_Level'),
            ('Order_Quantity', 'Stock_Level'),
            ('Demand', 'Stock_Level'),
            ('Stock_Level', 'Stockout'),
            ('Promo_Active', 'Demand'),
            ('Competitor_Promo', 'Demand'),
            ('Weather_Impact', 'Demand'),
            ('Seasonal_Factor', 'Demand'),
            ('Lead_Time', 'Stock_Level'),
            ('Days_Since_Last_Order', 'Order_Quantity'),
        ]
        
        self.causal_graph.add_edges_from(edges)
        return self.causal_graph
    
    def estimate_causal_effects(self, df: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Estimate causal effects using statistical methods"""
        try:
            if not DOWHY_AVAILABLE:
                return self._simple_causal_estimation(df, treatment, outcome)
            
            # DoWhy causal analysis
            model = CausalModel(
                data=df,
                treatment=treatment,
                outcome=outcome,
                graph=self._convert_to_dowhy_graph()
            )
            
            # Identify causal effect
            identified_estimand = model.identify_effect()
            
            # Estimate causal effect
            causal_estimate = model.estimate_effect(identified_estimand,
                                                  method_name="backdoor.linear_regression")
            
            return {
                'effect_size': causal_estimate.value,
                'confidence_interval': [causal_estimate.value - 1.96 * causal_estimate.stderr,
                                      causal_estimate.value + 1.96 * causal_estimate.stderr],
                'p_value': causal_estimate.p_value if hasattr(causal_estimate, 'p_value') else None
            }
        except Exception as e:
            print(f"Causal analysis error: {e}")
            return self._simple_causal_estimation(df, treatment, outcome)
    
    def _simple_causal_estimation(self, df: pd.DataFrame, treatment: str, outcome: str) -> Dict:
        """Simple causal effect estimation using linear regression"""
        # Control for confounders
        controls = ['Month', 'Quarter', 'Product_ID_encoded']
        controls = [c for c in controls if c in df.columns]
        
        formula_vars = [treatment] + controls
        X = df[formula_vars].fillna(0)
        y = df[outcome].fillna(0)
        
        model = LinearRegression()
        model.fit(X, y)
        
        treatment_effect = model.coef_[0]  # First coefficient is treatment effect
        
        return {
            'effect_size': treatment_effect,
            'confidence_interval': [treatment_effect * 0.8, treatment_effect * 1.2],  # Rough estimate
            'p_value': None
        }
    
    def _convert_to_dowhy_graph(self) -> str:
        """Convert NetworkX graph to DoWhy format"""
        edges_str = []
        for edge in self.causal_graph.edges():
            edges_str.append(f"{edge[0]} -> {edge[1]}")
        return "; ".join(edges_str)
    
    def interpret_causal_effect_with_gemini(treatment, outcome, effect_size, confidence_interval):
        prompt = f"""
            You're an AI business analyst assisting a supply chain operations team.

            Here is the result of a causal analysis:

            - Treatment variable: Supplier_Delay  
            - Outcome variable: Stockout  
            - Estimated causal effect: -0.073  
            - Confidence interval: [-0.058, -0.087]

            Explain these results in a clear, natural, and professional tone that a business stakeholder can understand — no technical jargon. Include:

            1. A one-sentence summary of the key finding.
            2. A brief explanation of why this result might be happening, even if it's counterintuitive.
            3. Three actionable recommendations based on the insight.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()


class StockoutPredictor:
    """Advanced stockout prediction with multiple models"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.scalers = {}
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for modeling"""
        # Remove rows with missing target
        df_clean = df.dropna(subset=['Stockout', 'Stock_Level'])
        
        # Select features for modeling
        feature_cols = [col for col in df_clean.columns if col not in [
            'Date', 'Product_ID', 'Stockout', 'Stock_Level'
        ] and not col.endswith('_encoded')]
        
        # Add encoded categorical features
        categorical_encoded = [col for col in df_clean.columns if col.endswith('_encoded')]
        feature_cols.extend(categorical_encoded)
        
        # Handle missing values
        X = df_clean[feature_cols].fillna(0)
        y_stockout = df_clean['Stockout']
        y_stock_level = df_clean['Stock_Level']
        
        return X, pd.DataFrame({'stockout': y_stockout, 'stock_level': y_stock_level})
    
    def train_stockout_classifier(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train stockout classification models"""
        print("Training stockout classification models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced')
        }
        
        results = {}
        
        for name, model in models.items():
            if name == 'LogisticRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            report = classification_report(y_test, y_pred, output_dict=True)
            
            results[name] = {
                'model': model,
                'accuracy': report['accuracy'],
                'precision': report['1']['precision'],
                'recall': report['1']['recall'],
                'f1': report['1']['f1-score'],
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'test_labels': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))
        
        self.models['stockout_classifiers'] = results
        self.scalers['stockout'] = scaler
        
        return results
    
    def train_stock_level_regressor(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train stock level regression models"""
        print("Training stock level regression models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42),
            'LightGBM': lgb.LGBMRegressor(random_state=42, verbose=-1),
            'LinearRegression': LinearRegression(),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            if name in ['LinearRegression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1))) * 100
            
            results[name] = {
                'model': model,
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'predictions': y_pred,
                'test_labels': y_test
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[f'{name}_regression'] = dict(zip(X.columns, model.feature_importances_))
        
        self.models['stock_level_regressors'] = results
        self.scalers['stock_level'] = scaler
        
        return results
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train all prediction models"""
        X, y = self.prepare_data(df)
        
        # Train classification models
        stockout_results = self.train_stockout_classifier(X, y['stockout'])
        
        # Train regression models
        stock_level_results = self.train_stock_level_regressor(X, y['stock_level'])
        
        return {
            'stockout_classification': stockout_results,
            'stock_level_regression': stock_level_results
        }

class CounterfactualSimulator:
    """Simulate counterfactual scenarios and interventions"""
    
    def __init__(self, predictor: StockoutPredictor):
        self.predictor = predictor
        
    def simulate_intervention(self, df: pd.DataFrame, intervention: Dict, 
                            model_type: str = 'XGBoost') -> pd.DataFrame:
        """Simulate the effect of an intervention"""
        df_intervention = df.copy()
        
        # Apply intervention
        for variable, change in intervention.items():
            if variable in df_intervention.columns:
                if isinstance(change, dict):
                    # Multiplicative change
                    if 'multiply' in change:
                        df_intervention[variable] *= change['multiply']
                    # Additive change
                    elif 'add' in change:
                        df_intervention[variable] += change['add']
                    # Set to specific value
                    elif 'set' in change:
                        df_intervention[variable] = change['set']
                else:
                    # Direct assignment
                    df_intervention[variable] = change
        
        # Prepare data for prediction
        X_original, _ = self.predictor.prepare_data(df)
        X_intervention, _ = self.predictor.prepare_data(df_intervention)
        
        # Get predictions
        results = {}
        
        # Stockout predictions
        if 'stockout_classifiers' in self.predictor.models:
            stockout_model = self.predictor.models['stockout_classifiers'][model_type]['model']
            
            if model_type == 'LogisticRegression':
                scaler = self.predictor.scalers['stockout']
                original_pred = stockout_model.predict_proba(scaler.transform(X_original))[:, 1]
                intervention_pred = stockout_model.predict_proba(scaler.transform(X_intervention))[:, 1]
            else:
                original_pred = stockout_model.predict_proba(X_original)[:, 1]
                intervention_pred = stockout_model.predict_proba(X_intervention)[:, 1]
            
            results['stockout_risk_original'] = original_pred
            results['stockout_risk_intervention'] = intervention_pred
            results['stockout_risk_change'] = intervention_pred - original_pred
        
        # Stock level predictions
        if 'stock_level_regressors' in self.predictor.models:
            stock_model = self.predictor.models['stock_level_regressors'][model_type]['model']
            
            if model_type == 'LinearRegression':
                scaler = self.predictor.scalers['stock_level']
                original_pred = stock_model.predict(scaler.transform(X_original))
                intervention_pred = stock_model.predict(scaler.transform(X_intervention))
            else:
                original_pred = stock_model.predict(X_original)
                intervention_pred = stock_model.predict(X_intervention)
            
            results['stock_level_original'] = original_pred
            results['stock_level_intervention'] = intervention_pred
            results['stock_level_change'] = intervention_pred - original_pred
        
        # Create results dataframe
        results_df = pd.DataFrame(results)
        results_df['Date'] = df['Date'].values
        results_df['Product_ID'] = df['Product_ID'].values
        
        return results_df
    
    def what_if_analysis(self, df: pd.DataFrame, scenarios: Dict) -> Dict:
        """Run multiple what-if scenarios"""
        scenario_results = {}
        
        for scenario_name, intervention in scenarios.items():
            print(f"Running scenario: {scenario_name}")
            results = self.simulate_intervention(df, intervention)
            scenario_results[scenario_name] = {
                'results': results,
                'avg_stockout_reduction': results['stockout_risk_change'].mean(),
                'avg_stock_increase': results['stock_level_change'].mean(),
                'total_stockouts_prevented': (results['stockout_risk_change'] < -0.1).sum()
            }
        
        return scenario_results

class SupplyChainDashboard:
    """Streamlit dashboard for supply chain analytics"""
    
    def __init__(self):
        st.set_page_config(
            page_title="Supply Chain Stockout Forecasting",
            page_icon="🚚",
            layout="wide"
        )
    
    def run_dashboard(self, df: pd.DataFrame, predictor: StockoutPredictor, 
                     simulator: CounterfactualSimulator,causal_analyzer):
        """Main dashboard interface"""
        
        st.title("🚚 Supply Chain Stockout Forecasting Dashboard")
        st.sidebar.title("Navigation")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Choose a page",
            ["Overview", "Forecasting", "Intervention Simulation", "Model Performance","Causal Analysis"]
        )
        
        if page == "Overview":
            self.overview_page(df)
        elif page == "Forecasting":
            self.forecasting_page(df, predictor)
        elif page == "Intervention Simulation":
            self.intervention_page(df, simulator)
        elif page == "Model Performance":
            self.performance_page(predictor)
        elif page == "Causal Analysis":
            self.causal_analysis_page(df,causal_analyzer)
    
    def overview_page(self, df: pd.DataFrame):
        """Overview dashboard page"""
        st.header("📊 Supply Chain Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_stockouts = df['Stockout'].sum()
            st.metric("Total Stockouts", total_stockouts)
        
        with col2:
            stockout_rate = (df['Stockout'].mean() * 100)
            st.metric("Stockout Rate", f"{stockout_rate:.1f}%")
        
        with col3:
            avg_stock = df['Stock_Level'].mean()
            st.metric("Avg Stock Level", f"{avg_stock:.0f}")
        
        with col4:
            products_at_risk = df[df['Stock_Level'] < 50]['Product_ID'].nunique()
            st.metric("Products at Risk", products_at_risk)
        
        # Time series plots
        st.subheader("Stock Level Trends")
        
        # Product selection
        selected_products = st.multiselect(
            "Select Products",
            df['Product_ID'].unique(),
            default=df['Product_ID'].unique()[:3]
        )
        
        if selected_products:
            fig = px.line(
                df[df['Product_ID'].isin(selected_products)],
                x='Date', y='Stock_Level', color='Product_ID',
                title="Stock Level Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def forecasting_page(self, df: pd.DataFrame, predictor: StockoutPredictor):
        """Forecasting dashboard page"""
        st.header("🔮 Stockout Forecasting")
        
        # Product and date selection
        col1, col2 = st.columns(2)
        
        with col1:
            selected_product = st.selectbox(
                "Select Product",
                df['Product_ID'].unique()
            )
        
        with col2:
            date_range = st.date_input(
                "Select Date Range",
                value=[df['Date'].min().date(), df['Date'].max().date()],
                min_value=df['Date'].min().date(),
                max_value=df['Date'].max().date()
            )
        
        # Filter data
        product_data = df[
            (df['Product_ID'] == selected_product) &
            (df['Date'] >= pd.to_datetime(date_range[0])) &
            (df['Date'] <= pd.to_datetime(date_range[1]))
        ].sort_values('Date')
        
        if len(product_data) > 0:
            # Display predictions if models are trained
            if 'stockout_classifiers' in predictor.models:
                st.subheader("Stockout Risk Predictions")
                
                # Get best performing model
                best_model_name = max(
                    predictor.models['stockout_classifiers'].keys(),
                    key=lambda x: predictor.models['stockout_classifiers'][x]['f1']
                )
                
                st.info(f"Using best model: {best_model_name}")
                
                # Prepare data for prediction
                X, _ = predictor.prepare_data(product_data)
                
                # Get predictions
                model = predictor.models['stockout_classifiers'][best_model_name]['model']
                
                if best_model_name == 'LogisticRegression':
                    scaler = predictor.scalers['stockout']
                    stockout_probs = model.predict_proba(scaler.transform(X))[:, 1]
                else:
                    stockout_probs = model.predict_proba(X)[:, 1]
                
                # Create prediction dataframe
                pred_df = product_data.copy()
                pred_df['Stockout_Risk'] = stockout_probs
                pred_df['Risk_Category'] = pd.cut(
                    stockout_probs,
                    bins=[0, 0.3, 0.6, 1.0],
                    labels=['Low', 'Medium', 'High']
                )
                
                # Plot stockout risk over time
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Stock Level', 'Stockout Risk'),
                    vertical_spacing=0.1
                )
                
                # Stock level plot
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['Date'],
                        y=pred_df['Stock_Level'],
                        mode='lines',
                        name='Stock Level',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Add stockout markers
                stockout_dates = pred_df[pred_df['Stockout'] == 1]['Date']
                if len(stockout_dates) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=stockout_dates,
                            y=[0] * len(stockout_dates),
                            mode='markers',
                            name='Actual Stockouts',
                            marker=dict(color='red', size=10, symbol='x')
                        ),
                        row=1, col=1
                    )
                
                # Stockout risk plot
                colors = ['green' if x == 'Low' else 'orange' if x == 'Medium' else 'red' 
                         for x in pred_df['Risk_Category']]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_df['Date'],
                        y=pred_df['Stockout_Risk'],
                        mode='markers+lines',
                        name='Stockout Risk',
                        marker=dict(color=colors),
                        line=dict(color='purple')
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(height=600, title=f"Stockout Analysis for {selected_product}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk summary
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_risk_days = (pred_df['Risk_Category'] == 'High').sum()
                    st.metric("High Risk Days", high_risk_days)
                
                with col2:
                    avg_risk = pred_df['Stockout_Risk'].mean()
                    st.metric("Average Risk", f"{avg_risk:.3f}")
                
                with col3:
                    next_stockout_risk = pred_df['Stockout_Risk'].iloc[-1] if len(pred_df) > 0 else 0
                    st.metric("Latest Risk Score", f"{next_stockout_risk:.3f}")
    
    def intervention_page(self, df: pd.DataFrame, simulator: CounterfactualSimulator):
        """Intervention simulation dashboard page"""
        st.header("🔄 Intervention Simulation")
        
        st.write("Simulate the impact of different supply chain interventions")
        
        # Intervention configuration
        st.subheader("Configure Intervention")
        
        col1, col2 = st.columns(2)
        
        with col1:
            intervention_type = st.selectbox(
                "Intervention Type",
                ["Reduce Supplier Delays", "Increase Order Quantities", "Reduce Demand", "Multiple Interventions"]
            )
        
        with col2:
            selected_products = st.multiselect(
                "Select Products (empty for all)",
                df['Product_ID'].unique(),
                default=[]
            )
        
        # Build intervention based on selection
        intervention = {}
        
        if intervention_type == "Reduce Supplier Delays":
            delay_reduction = st.slider(
                "Supplier Delay Reduction (%)",
                min_value=0, max_value=100, value=50
            )
            intervention['Supplier_Delay'] = {'multiply': (100 - delay_reduction) / 100}
        
        elif intervention_type == "Increase Order Quantities":
            order_increase = st.slider(
                "Order Quantity Increase (%)",
                min_value=0, max_value=100, value=20
            )
            intervention['Order_Quantity'] = {'multiply': 1 + order_increase / 100}
        
        elif intervention_type == "Reduce Demand":
            demand_reduction = st.slider(
                "Demand Reduction (%)",
                min_value=0, max_value=50, value=10
            )
            intervention['Demand'] = {'multiply': (100 - demand_reduction) / 100}
        
        elif intervention_type == "Multiple Interventions":
            st.write("Configure multiple interventions:")
            
            # Supplier delay
            if st.checkbox("Reduce Supplier Delays"):
                delay_reduction = st.slider("Delay Reduction (%)", 0, 100, 30)
                intervention['Supplier_Delay'] = {'multiply': (100 - delay_reduction) / 100}
            
            # Order quantity
            if st.checkbox("Increase Order Quantities"):
                order_increase = st.slider("Order Increase (%)", 0, 100, 15)
                intervention['Order_Quantity'] = {'multiply': 1 + order_increase / 100}
            
            # Lead time
            if st.checkbox("Reduce Lead Times"):
                lead_time_reduction = st.slider("Lead Time Reduction (%)", 0, 50, 20)
                intervention['Lead_Time'] = {'multiply': (100 - lead_time_reduction) / 100}
        
        # Run simulation
        if st.button("Run Simulation") and intervention:
            with st.spinner("Running intervention simulation..."):
                # Filter data if specific products selected
                sim_data = df.copy()
                if selected_products:
                    sim_data = sim_data[sim_data['Product_ID'].isin(selected_products)]
                
                # Run simulation
                results = simulator.simulate_intervention(sim_data, intervention)
                
                # Display results
                st.subheader("Simulation Results")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_stockout_reduction = results['stockout_risk_change'].mean()
                    st.metric(
                        "Avg Stockout Risk Reduction",
                        f"{avg_stockout_reduction:.3f}",
                        delta=f"{avg_stockout_reduction:.3f}"
                    )
                
                with col2:
                    avg_stock_increase = results['stock_level_change'].mean()
                    st.metric(
                        "Avg Stock Level Increase",
                        f"{avg_stock_increase:.1f}",
                        delta=f"{avg_stock_increase:.1f}"
                    )
                
                with col3:
                    stockouts_prevented = (results['stockout_risk_change'] < -0.1).sum()
                    st.metric("Stockouts Prevented", stockouts_prevented)
                
                with col4:
                    total_products = results['Product_ID'].nunique()
                    improved_products = (results.groupby('Product_ID')['stockout_risk_change'].mean() < 0).sum()
                    st.metric("Products Improved", f"{improved_products}/{total_products}")
                
                # Visualization
                st.subheader("Impact Analysis")
                
                # Risk reduction by product
                product_impact = results.groupby('Product_ID').agg({
                    'stockout_risk_change': 'mean',
                    'stock_level_change': 'mean'
                }).reset_index()
                
                fig = px.scatter(
                    product_impact,
                    x='stockout_risk_change',
                    y='stock_level_change',
                    hover_data=['Product_ID'],
                    title="Intervention Impact by Product",
                    labels={
                        'stockout_risk_change': 'Stockout Risk Change',
                        'stock_level_change': 'Stock Level Change'
                    }
                )
                fig.add_hline(y=0, line_dash="dash", line_color="gray")
                fig.add_vline(x=0, line_dash="dash", line_color="gray")
                st.plotly_chart(fig, use_container_width=True)
                
                # Time series comparison
                if len(selected_products) == 1:
                    product_results = results[results['Product_ID'] == selected_products[0]]
                    
                    fig = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=('Stockout Risk Comparison', 'Stock Level Comparison')
                    )
                    
                    # Stockout risk comparison
                    fig.add_trace(
                        go.Scatter(
                            x=product_results['Date'],
                            y=product_results['stockout_risk_original'],
                            mode='lines',
                            name='Original Risk',
                            line=dict(color='red')
                        ),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=product_results['Date'],
                            y=product_results['stockout_risk_intervention'],
                            mode='lines',
                            name='With Intervention',
                            line=dict(color='green')
                        ),
                        row=1, col=1
                    )
                    
                    # Stock level comparison
                    fig.add_trace(
                        go.Scatter(
                            x=product_results['Date'],
                            y=product_results['stock_level_original'],
                            mode='lines',
                            name='Original Stock',
                            line=dict(color='blue'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=product_results['Date'],
                            y=product_results['stock_level_intervention'],
                            mode='lines',
                            name='With Intervention',
                            line=dict(color='orange'),
                            showlegend=False
                        ),
                        row=2, col=1
                    )
                    
                    fig.update_layout(height=600, title=f"Intervention Impact Over Time - {selected_products[0]}")
                    st.plotly_chart(fig, use_container_width=True)
    
    def performance_page(self, predictor: StockoutPredictor):
        """Model performance dashboard page"""
        st.header("📈 Model Performance")
        
        if not predictor.models:
            st.warning("No models have been trained yet. Please train models first.")
            return
        
        # Stockout classification performance
        if 'stockout_classifiers' in predictor.models:
            st.subheader("Stockout Classification Models")
            
            # Performance metrics table
            perf_data = []
            for model_name, results in predictor.models['stockout_classifiers'].items():
                perf_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy']:.3f}",
                    'Precision': f"{results['precision']:.3f}",
                    'Recall': f"{results['recall']:.3f}",
                    'F1-Score': f"{results['f1']:.3f}"
                })
            
            st.table(pd.DataFrame(perf_data))
            
            # Feature importance
            st.subheader("Feature Importance")
            
            model_name = st.selectbox(
                "Select Model for Feature Importance",
                list(predictor.models['stockout_classifiers'].keys())
            )
            
            if model_name in predictor.feature_importance:
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in predictor.feature_importance[model_name].items()
                ]).sort_values('Importance', ascending=True).tail(15)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title=f"Top 15 Features - {model_name}"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Stock level regression performance
        if 'stock_level_regressors' in predictor.models:
            st.subheader("Stock Level Regression Models")
            
            # Performance metrics table
            perf_data = []
            for model_name, results in predictor.models['stock_level_regressors'].items():
                perf_data.append({
                    'Model': model_name,
                    'MAE': f"{results['mae']:.2f}",
                    'RMSE': f"{results['rmse']:.2f}",
                    'MAPE': f"{results['mape']:.2f}%"
                })
            
            st.table(pd.DataFrame(perf_data))
    
    def causal_analysis_page(self, df: pd.DataFrame, causal_analyzer: CausalAnalyzer):
        st.header("🧠 Causal Analysis")
    
        st.write("Estimate causal effects between key variables.")

        treatment = st.selectbox("Select Treatment Variable", [
        'Supplier_Delay', 'Order_Quantity', 'Promo_Active', 'Competitor_Promo'
        ])
    
        outcome = st.selectbox("Select Outcome Variable", [
        'Stockout', 'Stock_Level', 'Demand'
        ])
    
        if st.button("Run Causal Analysis"):
            with st.spinner("Estimating causal effect..."):
                result = causal_analyzer.estimate_causal_effects(df, treatment, outcome)
            
                if result:
                    st.success(f"Causal Effect of {treatment} on {outcome}: {result['effect_size']:.3f}")
                    st.write(f"Confidence Interval: {result['confidence_interval']}")
                    if result['p_value'] is not None:
                        st.write(f"p-value: {result['p_value']:.4f}")
                else:
                    st.warning("Could not estimate causal effect.")
            gemini_output = CausalAnalyzer.interpret_causal_effect_with_gemini(treatment,outcome,result['effect_size'],result['confidence_interval'])
            st.subheader("AI Interpretation")
            st.markdown(gemini_output)


def main():
    """Main execution function"""
    print("🚚 Supply Chain Stockout Forecasting System")
    print("=" * 50)
    
    # Initialize components
    data_generator = SupplyChainDataGenerator(n_products=5)
    feature_engineer = FeatureEngineer()
    causal_analyzer = CausalAnalyzer()
    
    # Generate and prepare data
    print("Generating synthetic supply chain data...")
    df = data_generator.generate_data()
    print(f"Generated {len(df)} records for {df['Product_ID'].nunique()} products")
    
    # Feature engineering
    print("\nEngineering features...")
    df_features = feature_engineer.engineer_features(df)
    
    # Build causal graph
    print("\nBuilding causal graph...")
    causal_graph = causal_analyzer.build_causal_graph()
    print(f"Causal graph built with {len(causal_graph.nodes)} nodes and {len(causal_graph.edges)} edges")
    
    # Train models
    print("\nTraining prediction models...")
    predictor = StockoutPredictor()
    model_results = predictor.train_all_models(df_features)
    
    # Print model performance
    print("\n📊 Model Performance Summary:")
    print("-" * 30)
    
    if 'stockout_classification' in model_results:
        print("Stockout Classification Models:")
        for model_name, results in model_results['stockout_classification'].items():
            print(f"  {model_name}: F1={results['f1']:.3f}, Accuracy={results['accuracy']:.3f}")
    
    if 'stock_level_regression' in model_results:
        print("\nStock Level Regression Models:")
        for model_name, results in model_results['stock_level_regression'].items():
            print(f"  {model_name}: MAE={results['mae']:.2f}, RMSE={results['rmse']:.2f}")
    
    # Causal analysis example
    print("\n🔍 Causal Analysis:")
    print("-" * 20)
    
    # Analyze effect of supplier delays on stockouts
    delay_effect = causal_analyzer.estimate_causal_effects(
        df_features, 'Supplier_Delay', 'Stockout'
    )
    print(f"Supplier Delay → Stockout Effect: {delay_effect['effect_size']:.3f}")
    
    # Analyze effect of order quantity on stock level
    order_effect = causal_analyzer.estimate_causal_effects(
        df_features, 'Order_Quantity', 'Stock_Level'
    )
    print(f"Order Quantity → Stock Level Effect: {order_effect['effect_size']:.3f}")
    
    # Counterfactual simulation
    print("\n🔄 Counterfactual Simulation:")
    print("-" * 30)
    
    simulator = CounterfactualSimulator(predictor)
    
    # Define intervention scenarios
    scenarios = {
        "Reduce_Supplier_Delays_50%": {'Supplier_Delay': {'multiply': 0.5}},
        "Increase_Orders_20%": {'Order_Quantity': {'multiply': 1.2}},
        "Combined_Intervention": {
            'Supplier_Delay': {'multiply': 0.7},
            'Order_Quantity': {'multiply': 1.15}
        }
    }
    
    # Run what-if analysis
    scenario_results = simulator.what_if_analysis(df_features.head(1000), scenarios)
    
    for scenario_name, results in scenario_results.items():
        print(f"\n{scenario_name}:")
        print(f"  Avg Stockout Risk Reduction: {results['avg_stockout_reduction']:.3f}")
        print(f"  Avg Stock Level Increase: {results['avg_stock_increase']:.1f}")
        print(f"  Stockouts Prevented: {results['total_stockouts_prevented']}")
    
    # Save models and data
    print("\n💾 Saving models and data...")
    
    # Save trained models
    joblib.dump(predictor, 'supply_chain_predictor.joblib')
    
    # Save processed data
    df_features.to_csv('supply_chain_data_processed.csv', index=False)
    
    print("Models and data saved successfully!")
    print("\n🚀 Launching Streamlit Dashboard...")
    causal_analyzer =  CausalAnalyzer()
    dashboard = SupplyChainDashboard()
    dashboard.run_dashboard(df_features,predictor,simulator,causal_analyzer)
    
    return df_features, predictor, simulator

# Additional utility functions

def load_saved_models():
    """Load previously saved models"""
    try:
        predictor = joblib.load('supply_chain_predictor.joblib')
        df = pd.read_csv('supply_chain_data_processed.csv')
        simulator = CounterfactualSimulator(predictor)
        return df, predictor, simulator
    except FileNotFoundError:
        print("No saved models found. Please run main() first.")
        return None, None, None

def run_streamlit_dashboard():
    """Run the Streamlit dashboard separately"""
    df, predictor, simulator = load_saved_models()
    
    if df is not None:
        dashboard = SupplyChainDashboard()
        dashboard.run_dashboard(df, predictor, simulator,causal_analyzer)
    else:
        st.error("No saved models found. Please run the main training script first.")

def evaluate_business_impact(scenario_results: Dict, current_stockout_cost: float = 1000,
                           current_holding_cost: float = 10) -> Dict:
    """Calculate business impact of interventions"""
    impact_analysis = {}
    
    for scenario_name, results in scenario_results.items():
        # Calculate cost savings
        stockouts_prevented = results['total_stockouts_prevented']
        stockout_cost_savings = stockouts_prevented * current_stockout_cost
        
        # Calculate additional holding costs
        avg_stock_increase = results['avg_stock_increase']
        additional_holding_cost = avg_stock_increase * current_holding_cost * 52  # Annual
        
        # Net benefit
        net_benefit = stockout_cost_savings - additional_holding_cost
        
        impact_analysis[scenario_name] = {
            'stockout_cost_savings': stockout_cost_savings,
            'additional_holding_cost': additional_holding_cost,
            'net_annual_benefit': net_benefit,
            'roi': (net_benefit / additional_holding_cost * 100) if additional_holding_cost > 0 else float('inf')
        }
    
    return impact_analysis

# Example usage and testing
if __name__ == "__main__":
    # Run the complete pipeline
    df, predictor, simulator = main()
    
    # Additional analysis
    print("\n💰 Business Impact Analysis:")
    print("-" * 30)
    
    # Define sample scenarios for business impact
    sample_scenarios = {
        "Supplier_Improvement": {'Supplier_Delay': {'multiply': 0.6}},
        "Inventory_Optimization": {'Order_Quantity': {'multiply': 1.25}}
    }
    
    # Run scenario analysis
    sample_results = simulator.what_if_analysis(df.head(500), sample_scenarios)
    
    # Calculate business impact
    business_impact = evaluate_business_impact(sample_results)
    
    for scenario, impact in business_impact.items():
        print(f"\n{scenario}:")
        print(f"  Annual Cost Savings: ${impact['stockout_cost_savings']:,.2f}")
        print(f"  Additional Holding Cost: ${impact['additional_holding_cost']:,.2f}")
        print(f"  Net Annual Benefit: ${impact['net_annual_benefit']:,.2f}")
        print(f"  ROI: {impact['roi']:.1f}%")
    
    print("\n✅ Supply Chain Forecasting System Complete!")
    print("📊 Run 'run_streamlit_dashboard()' to launch the interactive dashboard")
    print("💾 Models saved to 'supply_chain_predictor.joblib'")
    print("📈 Data saved to 'supply_chain_data_processed.csv'")