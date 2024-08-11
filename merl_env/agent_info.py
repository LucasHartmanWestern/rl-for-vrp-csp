#created by Santiago 05/07/2024

class agent_info():
    
    def __init__(self, index, chargers, routes, unique_chargers, unique_traffic):
        org_lat, org_long, dest_lat, dest_long = routes
        
        self.idx = index
        self.chargers = chargers
        self.org_lat  = org_lat
        self.org_long = org_long
        self.dest_lat = dest_lat
        self.dest_long= dest_long
        self.unique_chargers = unique_chargers
        self.unique_traffic  = unique_traffic