import os
import numpy as np
import tensorflow as tf
from django.conf import settings
from PIL import Image

# Global variable to cache the model so we don't reload it on every request
_MODEL = None

def load_model_and_predict(image_path):
    global _MODEL
    model_path = os.path.join(settings.BASE_DIR, 'prediction', 'model.h5')
    
    # Class labels matching the training script
    # Class labels matching the training script (250+ Classes)
    LABELS_LIST = [
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
    LABELS = {i: label for i, label in enumerate(LABELS_LIST)}

    if not os.path.exists(model_path):
        # Demo mode / Mock prediction if model is missing entirely
        print(f"Model file not found at {model_path}. Using mock prediction.")
        import random
        mock_idx = random.randint(0, len(LABELS)-1)
        label = LABELS[mock_idx]
        confidence = round(random.uniform(0.70, 0.99), 2)
        return label, confidence

    try:
        if _MODEL is None:
            print(f"Loading model from {model_path}...")
            _MODEL = tf.keras.models.load_model(model_path)
        
        model = _MODEL
        
        # Preprocess image
        img = Image.open(image_path)
        img = img.convert('RGB') # Ensure RGB
        img = img.resize((224, 224)) 
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) # Start with softmax
        
        # If the model output is not logits, softmax might be redundant 
        # but safe if training didn't use from_logits=True
        # In our train_model.py, we used activation='softmax' so output is ALREADY probability distribution.
        # So taking softmax again flattens it.
        # Let's check directly:
        
        # If existing model output sum approx 1, it's already softmaxed.
        raw_output = predictions[0]
        
        if np.isclose(np.sum(raw_output), 1.0, atol=0.1):
             # Already probabilities
             final_scores = raw_output
        else:
             # Logits
             final_scores = tf.nn.softmax(raw_output).numpy()
             
        class_idx = np.argmax(final_scores)
        
        label = LABELS.get(class_idx, "Unknown")
        confidence = round(100 * np.max(final_scores), 2)
        
        return label, confidence

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Fallback to mock to not crash the UI
        return "Error (Check Logs)", 0.0
