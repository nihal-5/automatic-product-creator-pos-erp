from __future__ import annotations

from .models import Supplier, SKU, Location, InventoryRecord, SaleRecord


def sample_suppliers() -> list[Supplier]:
    return [
        Supplier(
            supplier_id="SUP-RED",
            name="Red Label Supplier",
            lead_time_days=5,
            min_order_qty=50,
            price_band={"SKU-RED-TEA": 1.1, "SKU-MASALA": 1.5},
        ),
        Supplier(
            supplier_id="SUP-GROC",
            name="Grocer Partner",
            lead_time_days=7,
            min_order_qty=100,
            price_band={"SKU-RICE": 0.9, "SKU-PULSES": 1.2},
        ),
    ]


def sample_skus() -> list[SKU]:
    return [
        SKU(sku="SKU-RED-TEA", supplier_id="SUP-RED", case_size=10),
        SKU(sku="SKU-MASALA", supplier_id="SUP-RED", case_size=12),
        SKU(sku="SKU-RICE", supplier_id="SUP-GROC", case_size=20),
        SKU(sku="SKU-PULSES", supplier_id="SUP-GROC", case_size=24),
    ]


def sample_locations() -> list[Location]:
    return [
        Location(location_id="DC-EAST", kind="dc", capacity=5000, safety_stock=100),
        Location(location_id="STORE-NY", kind="store", capacity=800, safety_stock=40),
        Location(location_id="STORE-SF", kind="store", capacity=750, safety_stock=35),
        Location(location_id="ONLINE", kind="online", capacity=1200, safety_stock=60),
    ]


def sample_inventory() -> list[InventoryRecord]:
    return [
        InventoryRecord("SKU-RED-TEA", "DC-EAST", on_hand=320, inbound=50),
        InventoryRecord("SKU-RED-TEA", "STORE-NY", on_hand=60, inbound=0),
        InventoryRecord("SKU-RED-TEA", "STORE-SF", on_hand=40, inbound=0),
        InventoryRecord("SKU-RED-TEA", "ONLINE", on_hand=70, inbound=20),
        InventoryRecord("SKU-MASALA", "DC-EAST", on_hand=220, inbound=0),
        InventoryRecord("SKU-RICE", "DC-EAST", on_hand=900, inbound=150),
        InventoryRecord("SKU-PULSES", "DC-EAST", on_hand=430, inbound=0),
    ]


def sample_sales() -> list[SaleRecord]:
    # qty sold over last N days
    return [
        SaleRecord("SKU-RED-TEA", "STORE-NY", qty=120, days=14),
        SaleRecord("SKU-RED-TEA", "STORE-SF", qty=80, days=14),
        SaleRecord("SKU-RED-TEA", "ONLINE", qty=150, days=14),
        SaleRecord("SKU-MASALA", "STORE-NY", qty=60, days=14),
        SaleRecord("SKU-RICE", "DC-EAST", qty=400, days=14),
        SaleRecord("SKU-PULSES", "DC-EAST", qty=210, days=14),
    ]
