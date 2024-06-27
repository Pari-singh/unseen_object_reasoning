class_dict_cifar_combined = \
    {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9,
     'apple': 10, 'aquarium_fish': 11, 'baby': 12, 'bear': 13, 'beaver': 14, 'bed': 15, 'bee': 16, 'beetle': 17, 'bicycle': 18,
     'bottle': 19, 'bowl': 20, 'boy': 21, 'bridge': 22, 'bus': 23, 'butterfly': 24, 'camel': 25, 'can': 26, 'castle': 27,
     'caterpillar': 28, 'cattle': 29, 'chair': 30, 'chimpanzee': 31, 'clock': 32, 'cloud': 33, 'cockroach': 34, 'couch': 35,
     'crab': 36, 'crocodile': 37, 'cup': 38, 'dinosaur': 39, 'dolphin': 40, 'elephant': 41, 'flatfish': 42, 'forest': 43,
     'fox': 44, 'girl': 45, 'hamster': 46, 'house': 47, 'kangaroo': 48, 'keyboard': 49, 'lamp': 50, 'lawn_mower': 51,
     'leopard': 52, 'lion': 53, 'lizard': 54, 'lobster': 55, 'man': 56, 'maple_tree': 57, 'motorcycle': 58, 'mountain': 59,
     'mouse': 60, 'mushroom': 61, 'oak_tree': 62, 'orange': 63, 'orchid': 64, 'otter': 65, 'palm_tree': 66, 'pear': 67,
     'pickup_truck': 68, 'pine_tree': 69, 'plain': 70, 'plate': 71, 'poppy': 72, 'porcupine': 73, 'possum': 74, 'rabbit': 75,
     'raccoon': 76, 'ray': 77, 'road': 78, 'rocket': 79, 'rose': 80, 'sea': 81, 'seal': 82, 'shark': 83, 'shrew': 84,
     'skunk': 85, 'skyscraper': 86, 'snail': 87, 'snake': 88, 'spider': 89, 'squirrel': 90, 'streetcar': 91, 'sunflower': 92,
     'sweet_pepper': 93, 'table': 94, 'tank': 95, 'telephone': 96, 'television': 97, 'tiger': 98, 'tractor': 99, 'train': 100,
     'trout': 101, 'tulip': 102, 'turtle': 103, 'wardrobe': 104, 'whale': 105, 'willow_tree': 106, 'wolf': 107, 'woman': 108,
     'worm': 109}

class_dict_cifar100 = \
    {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7,
     'bicycle': 8, 'bottle': 9, 'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15,
     'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 'chair': 20, 'chimpanzee': 21, 'clock': 22,
     'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 'dinosaur': 29,
     'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36,
     'house': 37, 'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43,
     'lizard': 44, 'lobster': 45, 'man': 46, 'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50,
     'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 'palm_tree': 56, 'pear': 57,
     'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64,
     'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72,
     'shark': 73, 'shrew': 74, 'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79,
     'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86,
     'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 'tulip': 92, 'turtle': 93,
     'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}

superclasses_combined = {'vehicle': [0,1,8, 9, 18, 23, 51, 58, 68, 79, 91, 95, 99, 100] ,
                'vegetation': [10, 43, 57, 59, 61, 62, 63, 64, 66, 67,69, 70, 72,80,92,93, 102, 106],
                # 'animals': [2, 3, 4, 5, 6, 7, 11, 12, 13, 14, 16, 17, 21, 24, 25, 28, 29, 31, 34, 36, 37,39, 40, 41, 42,
                #             44, 45, 46, 48, 52, 53, 54, 55, 56, 60, 65, 73, 74, 75, 76, 82, 77, 83, 84, 85, 87,
                #             88, 89, 90, 98, 101, 103, 105, 107, 108, 109 ],
                'land_animals': [2, 3,4,5,6,7,12,13,14,16,17,21,24,25,28,29,31,34,39,41,44,45,
                                 46,48,52,53,54,55,56,60,65,73,74,75,76,84,85,87,88,89,90,98,107,108,109],
                'water_animals': [11, 36, 37,40, 42, 55, 77,82,83,101,103,105],
                'objects': [15, 19, 20, 22, 26, 30, 32, 35, 38, 49, 50, 71, 94, 96, 97, 104 ],
                'structures':[22, 27, 47, 86, 78, 33, 81]}

superclasses_kmeans = {'vehicle': [8, 13, 41, 48, 58, 69, 81, 85, 89, 90],
                       'vegetation': [0, 33, 47, 49, 51, 52, 53, 54, 56, 57, 59, 60, 62, 70, 82, 83, 92, 96],
                       # 'animals': [1, 2, 3, 4, 6, 7, 11, 14, 15, 18, 19, 21, 24, 26, 27, 29, 30, 31, 32, 34, 35,
                       #             36, 38, 42, 43, 44, 45, 46, 50, 55, 63, 64, 65, 66, 72, 67, 73, 74, 75, 77,
                       #             78, 79, 80, 88, 91, 93, 95, 97, 98, 99],
                       'land_animals': [2, 3, 4, 6, 7, 11, 14, 15, 18, 19, 21, 24, 29, 31, 34, 35, 36,
                                        38, 42, 43, 44, 45, 46, 50, 55, 63, 64, 65, 66, 74, 75, 77, 78,
                                        79, 80, 88, 97, 98, 99],
                       'water_animals': [1, 26, 27, 30, 32, 45, 67, 72, 73, 91, 93, 95],
                       'objects': [5, 9, 10, 12, 16, 20, 22, 25, 28, 39, 40, 61, 84, 86, 87, 94],
                       'structures': [12, 17, 37, 68, 76, 23, 71]}

superclasses_map = {'vehicle':0, 'vegetation':1, 'land_animals':2,
                  'water_animals':3, 'objects':4, 'structures':5}

class_to_superclass = {8: 'vehicle', 13: 'vehicle', 41: 'vehicle', 48: 'vehicle',
                  58: 'vehicle', 69: 'vehicle', 81: 'vehicle', 85: 'vehicle',
                  89: 'vehicle', 90: 'vehicle', 0: 'vegetation', 33: 'vegetation',
                  47: 'vegetation', 49: 'vegetation', 51: 'vegetation', 52: 'vegetation',
                  53: 'vegetation', 54: 'vegetation', 56: 'vegetation', 57: 'vegetation',
                  59: 'vegetation', 60: 'vegetation', 62: 'vegetation', 70: 'vegetation',
                  82: 'vegetation', 83: 'vegetation', 92: 'vegetation', 96: 'vegetation',
                  1: 'water_animals', 2: 'land_animals', 3: 'land_animals', 4: 'land_animals',
                  6: 'land_animals', 7: 'land_animals', 11: 'land_animals', 14: 'land_animals',
                  15: 'land_animals', 18: 'land_animals', 19: 'land_animals', 21: 'land_animals',
                  24: 'land_animals', 26: 'water_animals', 27: 'water_animals', 29: 'land_animals',
                  30: 'water_animals', 31: 'land_animals', 32: 'water_animals', 34: 'land_animals',
                  35: 'land_animals', 36: 'land_animals', 38: 'land_animals', 42: 'land_animals',
                  43: 'land_animals', 44: 'land_animals', 45: 'water_animals', 46: 'land_animals',
                  50: 'land_animals', 55: 'land_animals', 63: 'land_animals', 64: 'land_animals',
                  65: 'land_animals', 66: 'land_animals', 72: 'water_animals', 67: 'water_animals',
                  73: 'water_animals', 74: 'land_animals', 75: 'land_animals', 77: 'land_animals',
                  78: 'land_animals', 79: 'land_animals', 80: 'land_animals', 88: 'land_animals',
                  91: 'water_animals', 93: 'water_animals', 95: 'water_animals', 97: 'land_animals',
                  98: 'land_animals', 99: 'land_animals', 5: 'objects', 9: 'objects', 10: 'objects',
                  12: 'structures', 16: 'objects', 20: 'objects', 22: 'objects', 25: 'objects',
                  39: 'objects', 40: 'objects', 61: 'objects', 84: 'objects', 86: 'objects', 87: 'objects',
                  94: 'objects', 17: 'structures', 37: 'structures', 68: 'structures', 23: 'structures',
                  71: 'structures', 28: 'objects', 76: 'structures'}

superclass_sane = {
    'aquatic': ['mammals'  'beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    'food': ['containers'  'bottles', 'bowls', 'cans', 'cups', 'plates'],
    'fruit_and_vegetables':    ['apples', 'mushrooms', 'oranges', 'pears', 'sweet_peppers'],
    'household_electrical_devices':    ['clock', 'computer_keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores':    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_manmade_outdoor_things':    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes':    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_and_herbivores':  ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_sized_mammals':    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates':    ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':  ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles':    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals':   ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees':   ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1':  ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2':  ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
     }