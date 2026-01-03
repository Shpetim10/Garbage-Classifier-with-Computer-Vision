"""
knowledge_base.py
Recyclability information and rules for each garbage class
"""

class RecyclabilityKnowledgeBase:
    """
    Knowledge base system for garbage classification and recyclability
    """
    
    def __init__(self):
        self.knowledge = {
            'battery': {
                'recyclable': True,
                'category': 'Hazardous Waste',
                'recycling_instructions': [
                    'Never throw batteries in regular trash',
                    'Take to designated battery recycling centers',
                    'Many stores have battery drop-off boxes',
                    'Contains toxic materials - must be handled properly'
                ],
                'environmental_impact': 'High - contains heavy metals that pollute soil and water',
                'special_notes': 'Different types (alkaline, lithium, rechargeable) may have different disposal methods',
                'color_code': '#FF6B6B',  # Red - hazardous
                'icon': '🔋'
            },
            
            'biological': {
                'recyclable': True,
                'category': 'Compostable Waste',
                'recycling_instructions': [
                    'Add to compost bin or green waste collection',
                    'Can be composted at home if available',
                    'Keep separate from regular trash',
                    'Helps create nutrient-rich soil'
                ],
                'environmental_impact': 'Low - naturally biodegradable and beneficial when composted',
                'special_notes': 'Composting reduces methane emissions from landfills',
                'color_code': '#51CF66',  # Green - eco-friendly
                'icon': '🍂'
            },
            
            'brown-glass': {
                'recyclable': True,
                'category': 'Glass Recycling',
                'recycling_instructions': [
                    'Rinse container before recycling',
                    'Remove caps and lids',
                    'Can be recycled infinitely without quality loss',
                    'Place in glass recycling bin'
                ],
                'environmental_impact': 'Medium - recyclable but energy-intensive to process',
                'special_notes': 'Brown glass is often used for beverages to protect from light',
                'color_code': '#8B4513',  # Brown
                'icon': '🍺'
            },
            
            'cardboard': {
                'recyclable': True,
                'category': 'Paper Products',
                'recycling_instructions': [
                    'Flatten boxes to save space',
                    'Remove tape, labels, and staples if possible',
                    'Keep dry - wet cardboard cannot be recycled',
                    'Place in paper recycling bin'
                ],
                'environmental_impact': 'Low - highly recyclable and biodegradable',
                'special_notes': 'Can be recycled 5-7 times before fibers become too short',
                'color_code': '#D4A574',  # Tan
                'icon': '📦'
            },
            
            'clothes': {
                'recyclable': True,
                'category': 'Textile Waste',
                'recycling_instructions': [
                    'Donate if in good condition',
                    'Take to textile recycling centers',
                    'Some retailers offer take-back programs',
                    'Can be repurposed or upcycled'
                ],
                'environmental_impact': 'High - textile production is resource-intensive',
                'special_notes': 'Only 15% of textiles are recycled globally. Donation is best option.',
                'color_code': '#845EF7',  # Purple
                'icon': '👕'
            },
            
            'green-glass': {
                'recyclable': True,
                'category': 'Glass Recycling',
                'recycling_instructions': [
                    'Rinse container before recycling',
                    'Remove caps and lids',
                    'Can be recycled infinitely without quality loss',
                    'Place in glass recycling bin'
                ],
                'environmental_impact': 'Medium - recyclable but energy-intensive to process',
                'special_notes': 'Green glass is common for wine and beer bottles',
                'color_code': '#2F9E44',  # Dark green
                'icon': '🍾'
            },
            
            'metal': {
                'recyclable': True,
                'category': 'Metal Recycling',
                'recycling_instructions': [
                    'Rinse food containers',
                    'Aluminum and steel are highly recyclable',
                    'Check if your area accepts mixed metals',
                    'Place in metal recycling bin'
                ],
                'environmental_impact': 'Low - recycling saves 95% of energy vs new production',
                'special_notes': 'Metal can be recycled indefinitely without losing quality',
                'color_code': '#ADB5BD',  # Gray
                'icon': '🥫'
            },
            
            'paper': {
                'recyclable': True,
                'category': 'Paper Products',
                'recycling_instructions': [
                    'Keep dry and clean',
                    'Remove plastic windows from envelopes',
                    'Shredded paper needs special handling',
                    'Place in paper recycling bin'
                ],
                'environmental_impact': 'Low - very recyclable, saves trees',
                'special_notes': 'Recycling one ton of paper saves 17 trees',
                'color_code': '#F8F9FA',  # Light gray
                'icon': '📄'
            },
            
            'plastic': {
                'recyclable': 'Depends on type',
                'category': 'Plastic Waste',
                'recycling_instructions': [
                    'Check the recycling number (1-7) on the item',
                    'Rinse containers before recycling',
                    'Only types 1, 2, and 5 are commonly recycled',
                    'Remove caps and labels if possible'
                ],
                'environmental_impact': 'Very High - takes 450+ years to decompose, pollutes oceans',
                'special_notes': 'Only 9% of all plastic ever made has been recycled. Reduce usage when possible.',
                'color_code': '#4C6EF5',  # Blue
                'icon': '♻️'
            },
            
            'shoes': {
                'recyclable': True,
                'category': 'Textile/Rubber Waste',
                'recycling_instructions': [
                    'Donate if in wearable condition',
                    'Nike, Adidas and others have recycling programs',
                    'Some facilities can recycle into athletic surfaces',
                    'Separate leather, rubber, and fabric if possible'
                ],
                'environmental_impact': 'High - complex materials, difficult to recycle',
                'special_notes': 'Over 300 million pairs of shoes end up in landfills annually',
                'color_code': '#F59F00',  # Orange
                'icon': '👟'
            },
            
            'trash': {
                'recyclable': False,
                'category': 'General Waste',
                'recycling_instructions': [
                    'This item cannot be recycled',
                    'Dispose in general waste bin',
                    'Check if item can be repaired or repurposed first',
                    'Consider reducing future similar waste'
                ],
                'environmental_impact': 'High - goes to landfill, takes space, may not decompose',
                'special_notes': 'When in doubt, research proper disposal for your area',
                'color_code': '#868E96',  # Dark gray
                'icon': '🗑️'
            },
            
            'white-glass': {
                'recyclable': True,
                'category': 'Glass Recycling',
                'recycling_instructions': [
                    'Rinse container before recycling',
                    'Remove caps and lids',
                    'Can be recycled infinitely without quality loss',
                    'Most valuable glass type for recycling'
                ],
                'environmental_impact': 'Medium - recyclable but energy-intensive to process',
                'special_notes': 'Clear glass is the most recyclable as it can be made into any color',
                'color_code': '#F1F3F5',  # Very light gray
                'icon': '🥛'
            }
        }
    
    def get_info(self, class_name):
        """
        Get recyclability information for a specific class
        
        Args:
            class_name: Name of the garbage class
            
        Returns:
            Dictionary with recyclability information
        """
        return self.knowledge.get(class_name, {
            'recyclable': 'Unknown',
            'category': 'Unknown',
            'recycling_instructions': ['Information not available for this item'],
            'environmental_impact': 'Unknown',
            'special_notes': 'Please consult local recycling guidelines',
            'color_code': '#868E96',
            'icon': '❓'
        })
    
    def get_recyclability_status(self, class_name):
        """
        Get simple recyclability status
        
        Returns:
            String: 'Recyclable', 'Not Recyclable', 'Conditional', or 'Unknown'
        """
        info = self.get_info(class_name)
        recyclable = info.get('recyclable', 'Unknown')
        
        if recyclable is True:
            return 'Recyclable ♻️'
        elif recyclable is False:
            return 'Not Recyclable ⛔'
        elif isinstance(recyclable, str) and 'Depends' in recyclable:
            return 'Conditional ⚠️'
        else:
            return 'Unknown ❓'
    
    def get_all_categories(self):
        """
        Get list of all waste categories
        
        Returns:
            List of unique categories
        """
        categories = set()
        for item_info in self.knowledge.values():
            categories.add(item_info.get('category', 'Unknown'))
        return sorted(list(categories))
    
    def get_items_by_category(self, category):
        """
        Get all items in a specific category
        
        Args:
            category: Name of the category
            
        Returns:
            List of item names in that category
        """
        items = []
        for item_name, item_info in self.knowledge.items():
            if item_info.get('category') == category:
                items.append(item_name)
        return sorted(items)
    
    def get_environmental_score(self, class_name):
        """
        Get environmental impact score (0-100, lower is better)
        
        Args:
            class_name: Name of the garbage class
            
        Returns:
            Integer score
        """
        impact_map = {
            'Low': 20,
            'Medium': 50,
            'High': 80,
            'Very High': 95,
            'Unknown': 50
        }
        
        info = self.get_info(class_name)
        impact = info.get('environmental_impact', 'Unknown')
        
        # Extract impact level from the string
        for level, score in impact_map.items():
            if level in impact:
                return score
        
        return 50  # Default
    
    def compare_items(self, class_name1, class_name2):
        """
        Compare environmental impact of two items
        
        Returns:
            Dictionary with comparison results
        """
        score1 = self.get_environmental_score(class_name1)
        score2 = self.get_environmental_score(class_name2)
        
        if score1 < score2:
            better = class_name1
            worse = class_name2
        else:
            better = class_name2
            worse = class_name1
        
        return {
            'better_choice': better,
            'worse_choice': worse,
            'difference': abs(score1 - score2)
        }
    
    def get_quick_facts(self):
        """
        Get interesting quick facts about recycling
        
        Returns:
            List of fact strings
        """
        facts = [
            "🌍 Recycling one aluminum can saves enough energy to power a TV for 3 hours",
            "🌲 Recycling one ton of paper saves 17 trees and 7,000 gallons of water",
            "♻️ Glass can be recycled infinitely without losing quality",
            "🥤 Plastic takes up to 1,000 years to decompose in landfills",
            "📱 E-waste contains valuable materials like gold, silver, and copper",
            "🗑️ The average person generates 4.5 pounds of trash per day",
            "🌊 8 million tons of plastic enter the ocean every year",
            "♻️ Only 9% of all plastic waste has ever been recycled",
            "🔋 One recycled battery can prevent 1.5kg of soil pollution",
            "👕 The fashion industry is the 2nd largest polluter after oil"
        ]
        return facts
    
    def search_knowledge(self, query):
        """
        Search the knowledge base for a specific term
        
        Args:
            query: Search term
            
        Returns:
            List of matching items with relevance
        """
        query = query.lower()
        results = []
        
        for item_name, item_info in self.knowledge.items():
            relevance = 0
            
            # Check if query matches item name
            if query in item_name.lower():
                relevance += 10
            
            # Check category
            if query in item_info.get('category', '').lower():
                relevance += 5
            
            # Check instructions
            for instruction in item_info.get('recycling_instructions', []):
                if query in instruction.lower():
                    relevance += 2
            
            # Check notes
            if query in item_info.get('special_notes', '').lower():
                relevance += 2
            
            if relevance > 0:
                results.append({
                    'item': item_name,
                    'relevance': relevance,
                    'info': item_info
                })
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance'], reverse=True)
        return results


# Example usage and testing
if __name__ == "__main__":
    kb = RecyclabilityKnowledgeBase()
    
    # Test getting info
    print("Battery Info:")
    print(kb.get_info('battery'))
    
    print("\n\nAll Categories:")
    print(kb.get_all_categories())
    
    print("\n\nItems in Glass Recycling:")
    print(kb.get_items_by_category('Glass Recycling'))
    
    print("\n\nEnvironmental Comparison:")
    print(kb.compare_items('plastic', 'paper'))
    
    print("\n\nSearch for 'recycle':")
    results = kb.search_knowledge('recycle')
    for r in results[:3]:
        print(f"  {r['item']}: relevance {r['relevance']}")