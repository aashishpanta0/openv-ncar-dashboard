import numpy as np

class AIInsightsLogic:
    def __init__(self):
        # Simulate some climate data (replace with actual data in practice)
        self.data = np.random.rand(100, 100) * 100  # Simulating climate data

    def generate_insights(self):
        """Generate insights about the dataset."""
        # Calculate min and max values from the data
        min_val = np.nanmin(self.data)
        max_val = np.nanmax(self.data)

        # Create a message with insights
        insight_message = f"AI Insights:\n\n- Maximum Value: {max_val}\n- Minimum Value: {min_val}\n"
        
        return insight_message
