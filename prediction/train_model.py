import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import shutil

# Configuration
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 5 # Small number for synthetic/quick training

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.h5')

LABELS = [
    'Acanthosis Nigricans', 'Acne Conglobata', 'Acne Cosmetica', 'Acne Excoriee', 'Acne Fulminans', 
    'Acne Keloidalis Nuchae', 'Acne Mechanica', 'Acne Medicamentosa', 'Acne Miliaris Necrotica', 'Acne Vulgaris', 
    'Acne with Facial Edema', 'Acquired Digital Fibrokeratoma', 'Acquired Generalized Hypertrichosis', 
    'Acquired Ichthyosis', 'Acquired Perforating Dermatosis', 'Acrocyanosis', 'Acrodermatitis Continua of Hallopeau', 
    'Acrodermatitis Enteropathica', 'Acrokeratosis Paraneoplastica of Bazex', 'Acrokeratosis Verruciformis', 
    'Acroosteolysis', 'Acropustulosis of Infancy', 'Actinic Granuloma', 'Actinic Keratosis', 'Actinic Prurigo', 
    'Acute Generalized Exanthematous Pustulosis', 'Acute Paronychia', 'Addison Disease', 'Adverse Cutaneous Drug Reactions', 
    'Age Spots', 'Airborne Contact Allergic Dermatitis', 'Albinism', 'Allergic Reactions', 'Alopecia Areata', 
    'Alopecia Neoplastica', 'Anagen Effluvium', 'Androgenic Alopecia', 'Cicatricial Alopecia', 
    'Central Centrifugal Cicatricial Alopecia', 'Frontal Fibrosing Alopecia', 'Hot Comb Alopecia', 
    'Lipedematous Alopecia', 'Male Pattern Baldness', 'Noncicatricial Alopecia', 'Ophiasis', 'Pressure Alopecia', 
    'Pseudopelade of Brocq', 'Telogen Effluvium', 'Traction Alopecia', 'Triangular Alopecia', 'Tumor Alopecia', 
    'Alpha-1-antitrypsin Deficiency', 'Amputation Stump Neuroma', 'Amyloidosis Cutaneous', 'Anderson-Fabry Disease', 
    'Anetoderma', 'Angioedema', 'Angiofibromas', 'Angiokeratoma', 'Angioleiomyoma', 'Angiolipoma', 
    'Angiolymphoid Hyperplasia with Eosinophilia', 'Angioma', 'Angioma Serpiginosum', 'Angiosarcoma', 
    'Angular Cheilitis', 'Annular Erythema', 'Anthrax', 'Antiphospholipid Syndrome', 'Antisynthetase Syndrome', 
    'Aphthous Ulcers', 'Apocrine Carcinoma', 'Apocrine Hidrocystoma', 'Appendageal Tumours', 'Aquagenic Pruritus', 
    'Argyria', 'Arteriovenous Malformations Cutaneous', 'Ash-leaf Spots', 'Aspergillus', 'Atopic Dermatitis', 
    'Atopic Eruption of Pregnancy', 'Atrophic Glossitis', 'Atrophoderma of Pasini and Pierini', 
    'Atrophoderma Vermiculatam', 'Atypical Dysplastic Melanocytic Naevus', 'Atypical Fibroxanthoma', 
    'Atypical Mole Syndrome', 'Autoimmune Diseases', 'Autosomal Recessive Congenital Ichthyosis', 'Balanitis', 
    'Basal Cell Carcinoma', 'Beau Lines', 'Behcet Disease', 'Benign Skin Lesions', 'Birthmarks Pigmented', 
    'Birthmarks Red', 'Blistering Skin Conditions', 'Blue Nails', 'Blue Naevus', 'Body Lice', 'Boils', 
    'Bromidrosis', 'Bubble Hair Deformity', 'Bullae', 'Bullous Pemphigoid', 'Bullous Lupus Erythematosus', 
    'Burns', 'Cafe-au-lait Patch', 'Calcification of Skin', 'Calciphylaxis', 'Campbell de Morgan Spots', 
    'Candidal Infection', 'Canker Sore', 'Capillaritis', 'Capillary Malformations', 'Carbuncle', 
    'Carcinoid Syndrome', 'Carcinoma Cuniculatum', 'Cellulitis', 'Chafing', 'Chancroid', 'Chapped Hands', 
    'Cherry Angioma', 'Chevron Nail', 'Chickenpox', 'Chikungunya', 'Chilblains', 'Chilblain Lupus Erythematosus', 
    'Childhood Granulomatous Periorificial Dermatitis', 'Chloracne', 'Cholesterol Emboli', 
    'Chondrodermatitis Nodularis', 'Chromhidrosis', 'Chronic Actinic Dermatitis', 'Chronic Bullous Disease of Childhood', 
    'Chronic Hives', 'Chronic Paronychia', 'Chronic Superficial Scaly Dermatitis', 'Churg-Strauss Syndrome', 
    'Clear Cell Acanthoma', 'Clear Cell Hidradenoma', 'Clubbing', 'Cockade Naevus', 'Cold Sores', 
    'Combined Naevus', 'Comedo Naevus', 'Compulsive Skin Picking Disorder', 'Condylomata Acuminata', 
    'Confluent and Reticulate Papillomatosis', 'Congenital Dermal Melanocytosis', 'Congenital Melanocytic Naevus', 
    'Congenital Onychodysplasia', 'Connective Tissue Disorders', 'Contact Dermatitis', 'Contact Allergic Dermatitis', 
    'Corns and Callosities', 'Costal Fringe', 'COVID-19 Cutaneous Features', 'Cowden Syndrome', 
    'Cow Milk Protein Allergy', 'Coxsackie', 'Crab Lice', 'Creeping Eruption', 'Crohn Disease Cutaneous', 
    'Cryoglobulinaemic Vasculitis', 'Cryptococcus', 'Cryptosporidiosis', 'Cutaneous Horn', 'Cutaneous Larva Migrans', 
    'Cutaneous Leiomyoma', 'Cutaneous Lupus Erythematosus', 'Cutaneous Lymphoma', 'Cutaneous Metastases', 
    'Cutaneous Small Vessel Vasculitis', 'Cutaneous Vasculitis', 'Cutis Laxa', 'CYLD Cutaneous Syndrome', 
    'Cylindroma', 'Cyst', 'Cytomegalovirus', 'Darier Disease', 'Degos Disease', 'Delusion of Parasitosis', 
    'Dengue Fever', 'Dercum Disease', 'Dermal Hypersensitivity Reactions', 'Dermatitis Asteatotic', 
    'Dermatitis Artefacta', 'Dermatitis Herpetiformis', 'Dermatitis Nummular', 'Dermatitis Dyshidrotic', 
    'Eczema Coxsackium', 'Eczema Herpeticum', 'Eczema Gravitational', 'Eczema Hand and Foot', 'Eczema Napkin', 
    'Periorificial Dermatitis', 'Perioral Dermatitis', 'Periorbital Dermatitis', 'Seborrheic Dermatitis', 
    'Dermatofibroma', 'Dermatofibrosarcoma Protuberans', 'Dermatomyositis', 'Dermatosis Papulosa Nigra', 
    'Dermoid Cyst', 'Diabetic Dermopathy', 'Diffuse Large B-cell Lymphoma', 'Digital Mucous Cyst', 
    'Discoid Lupus Erythematosus', 'Dissecting Cellulitis', 'Disseminate and Recurrent Infundibulofolliculitis', 
    'Disseminated Superficial Actinic Porokeratosis', 'Drug Allergies', 'Drug Hypersensitivity Syndrome', 
    'Drug Rashes', 'Drug-induced Serum Sickness', 'Duhring Disease', 'Dyskeratosis Congenita', 'Dysplastic Naevus', 
    'Dyshidrosis', 'Eccrine Hidrocystoma', 'Eccrine Poroma', 'Ecthyma', 'Ectodermal Dysplasias', 
    'Ehlers-Danlos Syndrome', 'Elastosis Perforans Serpiginosa', 'Elephantiasis', 'En Coup de Sabre', 
    'ENA-positive Annular Erythema', 'Eosinophilic Annular Erythema', 'Eosinophilic Cellulitis', 
    'Eosinophilic Fasciitis', 'Eosinophilic Folliculitis', 'Epidermal Naevi', 'Epidermodysplasia Verruciformis', 
    'Epidermoid Cyst', 'Epidermolysis Bullosa', 'Epidermolysis Bullosa Acquisita', 'Epidermolytic Ichthyosis', 
    'Erosive Pustular Dermatosis of Scalp', 'Eruptive Vellus Hair Cysts', 'Eruptive Xanthomata', 'Erysipelas', 
    'Erysipeloid', 'Erythema Chronicum Migrans', 'Erythema Ab Igne', 'Erythema Annulare Centrifugum', 
    'Erythema Dyschromicum Perstans', 'Erythema Elevatum Diutinum', 'Erythema Gyratum Repens', 
    'Erythema Infectiosum', 'Erythema Multiforme', 'Erythema Nodosum', 'Erythrasma', 'Erythroderma', 
    'Erythrokeratoderma Variabilis', 'Erythromelalgia', 'Erythromelanosis Follicularis Faciei', 
    'Erythroplasia of Queyrat', 'Exanthems', 'Exercise-induced Vasculitis', 'Exogenous Ochronosis', 
    'Extramammary Paget Disease', 'Fixed Drug Eruption', 'Folliculitis', 'Folliculitis Decalvans', 
    'Folliculitis Nares Perforans', 'Gram-negative Folliculitis', 'Hot Tub Folliculitis', 'Perforating Folliculitis', 
    'Scalp Folliculitis', 'Tufted Folliculitis', 'Fox–Fordyce Disease', 'Fungal Nail Infection', 
    'Ganglion', 'Gardner Syndrome', 'Generalised Essential Telangiectasia', 'Generalized Congenital Hypertrichosis', 
    'Generalized Hyperhidrosis', 'Genital Herpes', 'Genital Leiomyoma', 'Genital Odema', 'Genital Psoriasis', 
    'Genital Warts', 'Geographic Tongue', 'Gianotti-Crosti Syndrome', 'Giant Cell Arteritis', 'Giant Comedone', 
    'Giant Congenital Nevus', 'Glandular Rosacea', 'Glomulovenous Malformation', 'Glomus Tumour', 'Gnathophyma', 
    'Gorlin Syndrome', 'Gougerot-Blum Syndrome', 'Gouty Tophi', 'Graft-versus-host Disease', 'Graham-Little Syndrome', 
    'Granular Parakeratosis', 'Granuloma Annulare', 'Granuloma Faciale', 'Granuloma Inguinale', 
    'Granulomatous Cheilitis', 'Granulomatous Dermatitis', 'Granulomatous Facial Dermatitis', 
    'Granulomatous Perioral Dermatitis', 'Granulomatous Rosacea', 'Granulosis Rubra Nasi', 'Graves Disease', 
    'Green Nails', 'Grover Disease', 'Gustatory Hyperhidrosis', 'Guttate Psoriasis', 'Hair Casts', 
    'Hair Follicle Nevus', 'Hairy Palms and Soles', 'Half and Half Nails', 'Halo Naevus', 'Hangnail', 
    'Hapalonychia', 'Hand Foot and Mouth Disease', 'Hemangioma', 'Hematidrosis', 'Henoch Schoenlein Purpura', 
    'Herpes Simplex', 'Herpes Zoster', 'Herpetic Whitlow', 'Hidradenitis Suppurativa', 'Hirsutism', 
    'Hook Nail', 'Hyperhidrosis', 'Hypereosinophilic Syndrome', 'Hypohidrosis', 'Ichthyosis Hystrix', 
    'Ichthyosis Vulgaris', 'Lamellar Ichthyosis', 'Id Reactions', 'Idiopathic Eruptive Macular Pigmentation', 
    'Idiopathic Facial Aseptic Granuloma', 'Idiopathic Guttate Hypomelanosis', 'IgA Pemphigus', 'Impetigo', 
    'Inclusion Body Myositis', 'Incontinentia Pigmenti', 'Infantile Digital Fibromatosis', 'Infantile Haemangiomas', 
    'Infectious Mononucleosis', 'Inflammatory Linear Verrucous Epidermal Naevus', 'Ink-spot Lentigo', 
    'Intermittent Hair–follicle Dystrophy', 'Intertrigo', 'Intraepidermal Neutrophilic IgA Dermatosis', 
    'Intraepidermal Squamous Cell Carcinoma', 'Intrahepatic Cholestasis of Pregnancy', 'Inverted Follicular Keratosis', 
    'Irritant Contact Dermatitis', 'Isotretinoin Reaction', 'Itch', 'Itching Purpura', 'Jessner Lymphocytic Infiltrate', 
    'Jewellery Allergy', 'Job Syndrome', 'Jock Itch', 'Juvenile Dermatomyositis', 'Juvenile Idiopathic Arthritis', 
    'Juvenile Plantar Dermatosis', 'Juvenile Spring Eruption', 'Juvenile Xanthogranuloma', 'Kaposi Sarcoma', 
    'Kawasaki Disease', 'Keloid Scar', 'Keratoacanthoma', 'Keratolysis Exfoliativa', 'Keratosis Obturans', 
    'Keratosis Pilaris', 'Keratosis Pilaris Atrophicans', 'Keratosis Pilaris Atrophicans Faciei', 'Kerion'
]

NUM_CLASSES = len(LABELS)

def create_synthetic_data(base_dir, num_images_per_class=5):
    """Generates synthetic images for testing pipeline."""
    print(f"Generating synthetic data in {base_dir}...")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    for i, label in enumerate(LABELS):
        class_dir = os.path.join(base_dir, label)
        os.makedirs(class_dir, exist_ok=True)
        
        # Generate random images
        for j in range(num_images_per_class):
            # Create a random noise image
            img_array = np.random.randint(0, 255, (IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Draw a colored rectangle based on class index acts as a "feature"
            # This helps the model actually learn something distinct even if it's noise
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            color = tuple(np.random.randint(0, 255, 3))
            draw.rectangle([10+i*10, 10, 50+i*10, 50], fill=color)
            
            img.save(os.path.join(class_dir, f"{label}_{j}.jpg"))
    print("Synthetic data generation complete.")

def build_model():
    """Builds a simple transfer learning model or custom CNN."""
    # Using MobileNetV2 for transfer learning (even for synthetic, it's a good placeholder)
    base_model = MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False # Freeze base model

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train():
    # Check if data exists
    train_dir = os.path.join(DATA_DIR, 'train')
    val_dir = os.path.join(DATA_DIR, 'val')
    
    using_synthetic = False
    
    if not os.path.exists(DATA_DIR) or not os.path.exists(train_dir):
        print(f"Dataset not found at {DATA_DIR}.")
        print("Switching to SYNTHETIC DATA mode to ensure model.h5 is created.")
        create_synthetic_data(train_dir, num_images_per_class=5)
        create_synthetic_data(val_dir, num_images_per_class=2)
        using_synthetic = True
    else:
        print(f"Found dataset at {DATA_DIR}. Training on real data.")

    # Data Generators
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    print(f"Loading data from {train_dir}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=LABELS # Ensure consistent class ordering
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=LABELS
    )

    model = build_model()
    model.summary()

    print("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS if not using_synthetic else 3, # Fewer epochs for synthetic
        validation_data=validation_generator
    )

    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Model saved successfully.")
    
    if using_synthetic:
        print("\n" + "="*50)
        print("WARNING: MODEL TRAINED ON SYNTHETIC DATA")
        print(" This model allows the application to run but will NOT make accurate predictions.")
        print(" To train a real model:")
        print(f" 1. Delete the synthetic 'data' folder at {DATA_DIR}")
        print(f" 2. Place your 'train' and 'val' folders in {DATA_DIR}")
        print(" 3. Run this script again.")
        print("="*50 + "\n")

if __name__ == '__main__':
    train()
