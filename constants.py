#!/usr/bin/env python3

FEATURE_PER_CATEGORY = {
    'television':   ['brand', 'rating', 'size', 'price', 'shipping', 'condition', 'specific_feature'],
    'vacuum':       ['brand', 'rating', 'size', 'price', 'shipping', 'color', 'specific_feature'],
    'diapers':      ['brand', 'rating', 'size', 'price', 'shipping', 'specific_feature'],
    'smartphone':   ['brand', 'rating', 'size', 'price', 'shipping', 'condition', 'color', 'specific_feature'],
    'sofa':         ['brand', 'rating', 'size', 'price', 'shipping', 'color', 'specific_feature']
}

BRAND = {
    'television': ['Samsung', 'LG', 'Sony', 'Visio', 'TCL', 'Hisense', 'Panasonic', 'Sharp', 'Toshiba', 'Philips'],
    'vacuum': ['Dyson', 'Shark', 'Bissell', 'Hoover', 'Miele', 'Eureka', 'Oreck'],
    'diapers': ['Pampers', 'Huggies', 'Luvs', 'Seventh Generation', 'Honest Company'],
    'smartphone': ['Apple iPhone 14', 'Apple iPhone 14 Pro', 'Apple iPhone 14 Pro Max', 'Apple iPhone 13',
                    'Samsung Galaxy S22', 'Samsung Galaxy S22+', 'Samsung Galaxy S22 Ultra',
                    'Samsung Galaxy Z Flip 4', 'Samsung Galaxy Z Fold 4', 'Google Pixel 7',
                    'Google Pixel 7 Pro', 'Google Pixel 6a', 'Motorola Moto G Stylus (2022)',
                    'OnePlus 10 Pro', 'Xiaomi Redmi Note 11', 'Xiaomi Redmi Note 11 Pro'],
    'sofa': ['Homfa', 'LuxuryGoods', 'La-Z-Boy', 'Crate & Barrel', 'IKEA', 'Pottery Barn', 'West Elm', 'JoyBird']
}

PRICE = {
    'television': [100, 1000],
    'vacuum': [50, 500],
    'diapers': [15, 90],
    'smartphone': [100, 1100],
    'sofa': [250, 1800]
}

IMAGE = {
    'television': ['tv1.jpg', 'tv2.jpg', 'tv3.jpg', 'tv4.jpg', 'tv5.jpg', 'tv6.jpg', 'tv7.jpg', 'tv8.jpg', 'tv9.jpg', 'tv10.jpg'],
    'vacuum': ['vacuum1.jpg', 'vacuum2.jpg', 'vacuum3.jpg', 'vacuum4.jpg', 'vacuum5.jpg', 'vacuum6.jpg', 'vacuum7.jpg', 'vacuum8.jpg', 'vacuum9.jpg', 'vacuum10.jpg'],
    'diapers': ['diapers1.jpg', 'diapers2.jpg', 'diapers3.jpg', 'diapers4.jpg', 'diapers5.jpg', 'diapers6.jpg', 'diapers7.jpg', 'diapers8.jpg', 'diapers9.jpg', 'diapers10.jpg'],
    'smartphone': ['phone1.jpg', 'phone2.jpg', 'phone3.jpg', 'phone4.jpg', 'phone5.jpg', 'phone6.jpg', 'phone7.jpg', 'phone8.jpg', 'phone9.jpg', 'phone10.jpg'],
    'sofa': ['sofa1.jpg', 'sofa2.jpg', 'sofa3.jpg', 'sofa4.jpg', 'sofa5.jpg', 'sofa6.jpg', 'sofa7.jpg', 'sofa8.jpg', 'sofa9.jpg', 'sofa10.jpg']
}

SIZE = {
    'television': ['screen_size', 'inches', 29, 85],
    'vacuum': ['weight', 'pounds', 5, 25],
    'diapers': ['size', 'count', 20, 135],
    'smartphone': ['screen_size', 'inches', 5, 7],
    'sofa': ['length', 'meters', 1, 3]
}
COLOR = {
    'vacuum': [('black', 0.8), ('blue', 0.06), ('white', 0.05), ('gray', 0.03), ('purple', 0.03), ('red', 0.03)],
    'smartphone': [('black', 0.3), ('white', 0.2), ('blue', 0.1), ('pink', 0.1), ('green', 0.1),
                    ('red', 0.1), ('yellow', 0.1)],
    'sofa': [('black', 0.3), ('brown', 0.3), ('gray', 0.2), ('white', 0.05), ('red', 0.03), ('lilac', 0.03),
                ('purple', 0.03), ('teal', 0.03), ('pink', 0.03)]
}
FEATURE = {
    'television': {
        'binary': ['Touchscreen', 'Batteries Included', 'Wireless Connectivity'],
        'special': {
            'Resolution': ['720p (1280 x 720 pixels)', '1080p (1920 x 1080 pixels)',
                            '1440p (2560 x 1440 pixels)', '4K UHD (3840 x 2160 pixels)', '8K UHD (7680 × 4320 pixels)'],
            'Refresh Rate': ['60Hz', '120Hz', '240Hz', '480Hz', 'Adaptive (120/240)']
    }}, 
    'vacuum': {
        'binary': ['Bagless', 'Cordless', 'Washable Filter'],
        'special': {
            'Cord Length': ['10ft', '15ft', '20ft', '25ft', '30ft'],
            'Recommended Surface': ['carpet', 'hardwood', 'tile', 'all-surface', 'N/A']
    }}, 
    'diapers': {
        'binary': ['Hypoallergenic', 'Pull-ups', 'Biodegradable'],
        'special': {
            'Scent': ['unscented', 'floral', 'baby powder', 'citrus', 'fresh linen'],
            'Age Group': ['Newborn (Up to 10 lbs)', 'Size 1 (8-14 lbs)', 'Size 2 (12-18 lbs)', 'Size 3 (16-28 lbs)',
                            'Size 4 (22-37 lbs)', 'Size 5 (Over 27 lbs)', 'Size 6 (Over 35 lbs)']
    }}, 
    'smartphone': {
        'binary': ['Water Resistant', 'Unlocked', 'Dual SIM'],
        'special': {
            'Camera Megapixels': ['5MP', '8MP', '9MP', '11MP', '12MP', '16MP'],
            'Battery Capacity': ['2000mAh', '3000mAh', '4000mAh', '5000mAh', '6000mAh']
    }}, 
    'sofa': {
        'binary': ['Reclining', 'Sleeper', 'Cup Holders'],
        'special': {
            'Material': ['leather', 'microfiber', 'polyester', 'linen', 'velvet', 'upholstered'],
            'Weight capacity': ['250lbs', '500lbs', '750lbs', '1000lbs', '1250lbs']
    }}
}

OTHER = {
    'shipping': ['Free', 'Standard', 'Overnight'],  # Shipping Availability
    'condition': {
        'television': ['New', 'Like New', 'Good', 'OK', 'Poor'],
        'smartphone': ['New', 'Used', 'Refurbished'],
    }
}