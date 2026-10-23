"""
Diagnostic Morphological Knowledge Base for Elasmobranch Identification.

Maps species and species contrasts to specific photographic angles and anatomical landmarks
to guide field enumerators when identification confidence is low or ambiguous.
"""

from typing import List, Dict, Tuple, Optional

# Standard photo angle definitions from Kobo SHARC survey protocol
PHOTO_ANGLES = {
    4: "Full Dorsal View (Entire body from above, flat, including tail)",
    5: "Full Ventral View (Entire underside showing mouth, gill slits, and cloaca)",
    6: "Head & Snout (Dorsal close-up of snout, eyes, and spiracles)",
    7: "Mouth & Nasal Flaps (Ventral close-up of jaws, teeth, and nostrils)",
    8: "Dorsal Fin(s) & Spine (Close-up of fin shape, color markings, and spine if present)",
    9: "Pectoral & Pelvic Fins (Close-up of wing/fin edges and claspers if male)",
    10: "Tail & Caudal Fin (Close-up of tail folds, sting/spine, and caudal lobes)",
    11: "Skin / Denticle / Pattern Close-up (Macro shot of spots, stripes, or skin texture)",
    12: "Lateral Profile (Side view showing gill slits and body height)"
}

# Species diagnostic anatomical keys and primary angles
SPECIES_DIAGNOSTICS: Dict[str, Dict] = {
    "Squalus mahia": {
        "common_name": "Mahia Dogfish Shark",
        "type": "shark",
        "primary_angles": [
            "Dorsal fin spines (both 1st and 2nd dorsal fins with sharp spines)",
            "Head/snout ventral view (distance from snout tip to mouth)",
            "Lateral view (5 gill slits, lack of anal fin)"
        ],
        "key_features": "White-edged dorsal fins with stout grooved spines; no anal fin; distinctive slender snout."
    },
    "Pliotrema annae": {
        "common_name": "Anna's Sixgill Sawshark",
        "type": "shark_like_ray",
        "primary_angles": [
            "Lateral view of gill area (verify 6 gill openings instead of 5)",
            "Rostrum / saw close-up (position of nasal barbels relative to snout tip and mouth)",
            "Ventral mouth view (small teeth row)"
        ],
        "key_features": "6 pairs of gill slits; long blade-like saw rostrum with rostral teeth and two sensory barbels."
    },
    "Galeocerdo cuvier": {
        "common_name": "Tiger Shark",
        "type": "shark",
        "primary_angles": [
            "Flank/lateral view (dark vertical bars or spots along sides)",
            "Mouth/tooth close-up (heavily serrated cockscomb teeth)",
            "Head dorsal view (broad, blunt, square-shaped snout)"
        ],
        "key_features": "Square-shaped snout, curved cockscomb teeth with heavy serrations, prominent caudal keels, juvenile/adult flank bars."
    },
    "Maculabatis ambigua": {
        "common_name": "Baraka's Whipray",
        "type": "ray",
        "primary_angles": [
            "Dorsal disc pattern (disc shape, brown/grey coloration, presence of subtle speckles/spots)",
            "Tail base and sting (check for dorsal or ventral skin folds on tail)",
            "Ventral disc and mouth (snout tip angle and nasal curtain)"
        ],
        "key_features": "Rhomboidal disc, whip-like tail longer than disc without lower skin fold; dorsal disc plain brown/grey or with faint speckles."
    },
    "Pateobatis fai": {
        "common_name": "Pink Whipray",
        "type": "ray",
        "primary_angles": [
            "Dorsal disc color & texture (uniform pinkish, brownish or grayish with rough granule center)",
            "Tail base and post-sting (slender whip tail, no ventral skin fold)",
            "Snout angle (broadly rounded disc with slightly protruding snout tip)"
        ],
        "key_features": "Broad disc with rounded pectoral apex, uniform pinkish-buff to grey, very long whip tail without skin folds."
    },
    "Himantura uarnak": {
        "common_name": "Reticulate / Coachwhip Ray",
        "type": "ray",
        "primary_angles": [
            "Dorsal disc pattern (close-up of leopard-like dark spots, vermiculations, or reticulated honeycomb rings)",
            "Tail banding (alternating black and white rings on tail behind spine)"
        ],
        "key_features": "High-contrast leopard spots/rings across dorsal disc; alternating black-and-white banded tail."
    },
    "Taeniurops meyeni": {
        "common_name": "Round Ribbontail Ray / Blotched Stingray",
        "type": "ray",
        "primary_angles": [
            "Tail ventral view (broad, tall ventral cutaneous skin fold extending to tip of tail)",
            "Dorsal disc (circular disc with black and grey blotched or mottled pattern)"
        ],
        "key_features": "Thick circular disc, prominent tall black skin fold along ventral side of tail right to the tip."
    },
    "Neotrygon caeruleopunctata": {
        "common_name": "Bluespotted Maskray",
        "type": "ray",
        "primary_angles": [
            "Dorsal disc close-up (bright azure blue spots on yellowish/brown disc)",
            "Head dorsal view (dark transverse mask across the eyes)",
            "Tail (banded tail with short skin folds)"
        ],
        "key_features": "Striking bright blue spots on dorsal disc; distinctive dark mask over the eyes."
    },
    "Bathytoshia lata": {
        "common_name": "Brown Stingray",
        "type": "ray",
        "primary_angles": [
            "Tail ventral and dorsal folds (deep ventral skin fold, presence of row of tubercles/thorns along midline)",
            "Dorsal disc (dark brown/blackish, thick diamond disc, prominent mid-dorsal thorns)"
        ],
        "key_features": "Large heavy dark disc, sharp thorns along midline of back and tail, tail ventral fold."
    },
    "Rhinoptera jayakari": {
        "common_name": "Oman Cownose Ray",
        "type": "ray",
        "primary_angles": [
            "Head frontal view (bilobed snout with deep midline indentation)",
            "Lateral head view (high domed head distinct from pectoral wings)",
            "Mouth/tooth plates (crushing pavement tooth plates)"
        ],
        "key_features": "Distinctive double-lobed (bilobed) forehead; angular kite-shaped pectoral wings; high domed head."
    },
    "Mobula kuhlii": {
        "common_name": "Shortfin Pygmy Devil Ray",
        "type": "ray",
        "primary_angles": [
            "Frontal head view (short cephalic fins/horns rolled at sides of subterminal mouth)",
            "Dorsal fin (white tip on dorsal fin)",
            "Pectoral fins and disc width (broad wingspan, dark greenish-grey dorsal)"
        ],
        "key_features": "Cephalic horn fins flanking the mouth; subterminal mouth; white-tipped dorsal fin; short tail without spine."
    }
}

# Specific pairwise contrast guidance when model is uncertain between two species
PAIRWISE_DIAGNOSTICS: Dict[Tuple[str, str], str] = {
    ("Maculabatis ambigua", "Pateobatis fai"): (
        "Disc color & tail check: Submit a clear dorsal photo showing whole disc color/texture and a close-up of the tail base. "
        "Pateobatis fai has a uniform pinkish/brownish disc with a very rough central band of granules, "
        "whereas Maculabatis ambigua has a more oval/rhomboidal disc with smooth or fine denticles and differing snout angle."
    ),
    ("Maculabatis ambigua", "Himantura uarnak"): (
        "Dorsal pattern & tail banding: Himantura uarnak displays distinct dark leopard spots/reticulated rings and a strongly black-and-white banded tail. "
        "Capture a top-down shot of the disc surface and the tail section behind the spine."
    ),
    ("Taeniurops meyeni", "Bathytoshia lata"): (
        "Tail skin fold: Submit a close-up photo of the underside of the tail. "
        "Taeniurops meyeni possesses a deep, tall ventral skin fold that extends all the way to the tail tip, "
        "while Bathytoshia lata has a lower tail fold and midline dorsal thorns."
    ),
    ("Neotrygon caeruleopunctata", "Himantura uarnak"): (
        "Head mask & spot type: Neotrygon caeruleopunctata has bright blue round spots with a dark mask across the eyes. "
        "Submit a close-up photo of the head/eyes and disc pattern."
    ),
    ("Pliotrema annae", "Squalus mahia"): (
        "Gill slits and rostrum: Pliotrema annae is a sawshark with an elongated toothed rostrum and 6 gill slits. "
        "Squalus mahia has standard 5 gill slits and a regular snout. Submit a lateral photo of the gill area."
    ),
    ("Rhinoptera jayakari", "Mobula kuhlii"): (
        "Head & cephalic fins: Rhinoptera jayakari has a fleshy bilobed forehead with no rolled horns. "
        "Mobula kuhlii has distinct rolled cephalic horn-fins protruding forward. Submit a frontal head shot."
    )
}

def get_recommendation_for_prediction(
    top1_species: str,
    top1_conf: float,
    top2_species: Optional[str] = None,
    top2_conf: Optional[float] = 0.0,
    confidence_tier: str = "Uncertain"
) -> Dict[str, any]:
    """
    Generate actionable photo angle recommendations and diagnostic tips based on prediction confidence.
    """
    rec = {
        "status": confidence_tier,
        "message": "",
        "suggested_angles": [],
        "diagnostic_notes": ""
    }

    if confidence_tier == "High Confidence":
        rec["message"] = f"Identification highly consistent with {top1_species}."
        if top1_species in SPECIES_DIAGNOSTICS:
            rec["diagnostic_notes"] = SPECIES_DIAGNOSTICS[top1_species]["key_features"]
        return rec

    # Moderate or Uncertain: Generate actionable photo request
    margin = (top1_conf - top2_conf) if top2_species else top1_conf

    # Check for specific pairwise confusion
    pair_key = (top1_species, top2_species)
    rev_key = (top2_species, top1_species)

    if top2_species and pair_key in PAIRWISE_DIAGNOSTICS:
        rec["message"] = f"Ambiguous between {top1_species} ({top1_conf*100:.1f}%) and {top2_species} ({top2_conf*100:.1f}%)."
        rec["diagnostic_notes"] = PAIRWISE_DIAGNOSTICS[pair_key]
    elif top2_species and rev_key in PAIRWISE_DIAGNOSTICS:
        rec["message"] = f"Ambiguous between {top1_species} ({top1_conf*100:.1f}%) and {top2_species} ({top2_conf*100:.1f}%)."
        rec["diagnostic_notes"] = PAIRWISE_DIAGNOSTICS[rev_key]
    else:
        rec["message"] = (
            f"Prediction confidence for {top1_species} is {top1_conf*100:.1f}%. "
            f"Additional anatomical angles are recommended to confirm identification."
        )
        if top1_species in SPECIES_DIAGNOSTICS:
            rec["diagnostic_notes"] = SPECIES_DIAGNOSTICS[top1_species]["key_features"]

    # Provide recommended angles
    angles = []
    if top1_species in SPECIES_DIAGNOSTICS:
        angles.extend(SPECIES_DIAGNOSTICS[top1_species]["primary_angles"])
    if top2_species and top2_species in SPECIES_DIAGNOSTICS:
        for a in SPECIES_DIAGNOSTICS[top2_species]["primary_angles"]:
            if a not in angles:
                angles.append(a)

    rec["suggested_angles"] = angles[:3]
    return rec
