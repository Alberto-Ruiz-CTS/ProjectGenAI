from pydantic import BaseModel, Field

class Ticket(BaseModel):
    """TicketDatamodel"""
    name : str = Field(default = 'NA', description = "Name of the establishment where the purchase was made")
    address : str = Field(default = 'NA', description = "Address of the establishment")
    date : str = Field(default = 'NA', description = "Date of the ticket")
    total: str = Field(default = 'NA', description = "Total money amount of the ticket")
    category: str = Field(default = 'NA', description = "Choose one of the following: restaurant, groceries, leisure, gas, other")