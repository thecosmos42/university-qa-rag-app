"""
Program Extractor Module

Extracts which master's program(s) a user query is targeting.
This runs as a lightweight pre-retrieval step to scope hybrid_search
to only the relevant program(s) instead of searching across all documents.

Design principles:
- Pure rule-based alias matching — zero LLM calls, deterministic, fast
- Returns a list of program_names matching the normalised DB column
- Extend KNOWN_PROGRAMS to add more programs as new PDFs are ingested
- program_name values MUST match the snake_case DB column produced by the pipeline:
    [DocType]_[ProgramType]_[CourseName].pdf → program_name = camel_to_snake(CourseName)
    e.g. ArtificialIntelligence → artificial_intelligence

Example:
    extractor = ProgramExtractor()
    extractor.extract("What's the exam retake policy for MS AI?")
    # ["artificial_intelligence"]

    extractor.extract("Compare MS AI and masters econometrics")
    # ["artificial_intelligence", "econometrics_and_operations_research"]

    extractor.extract("What are the graduation requirements?")
    # []  ← ambiguous, caller should search all programs
"""

from dataclasses import dataclass, field


@dataclass
class ProgramFilter:
    """Result of program extraction from a user query."""
    program_names: list[str] = field(default_factory=list)
    """
    Normalised program_name values matching the DB column.
    e.g. ["artificial_intelligence"] or ["artificial_intelligence", "econometrics_and_operations_research"]
    Empty list means no specific program detected → search all programs.
    """


# ---------------------------------------------------------------------------
# Known programs registry
# Each entry: (program_name, [aliases])
#   - program_name must be the CourseName segment from the filename convention,
#     lowercased and stripped of spaces.
#   - aliases are lowercase strings that might appear in a user query.
# ---------------------------------------------------------------------------
KNOWN_PROGRAMS: list[tuple[str, list[str]]] = [
    # ── Computer Science & AI ────────────────────────────────────────────────
    (
        "artificial_intelligence",
        [
            "artificial intelligence", "ai",
            "ms ai", "msc ai", "m.sc. ai", "msc in ai",
            "master ai", "master's ai", "masters ai",
            "master artificial intelligence", "masters artificial intelligence",
            "msc artificial intelligence", "master of artificial intelligence",
        ],
    ),
    (
        "computer_science",
        [
            "computer science", "cs",
            "ms cs", "msc cs", "msc computer science",
            "master computer science", "masters computer science",
            "computer science joint degree", "joint degree computer science",
        ],
    ),
    (
        "computational_science",
        [
            "computational science",
            "msc computational science", "master computational science",
            "masters computational science", "computational science joint degree",
        ],
    ),
    (
        "computer_security",
        [
            "computer security", "cybersecurity", "cyber security",
            "msc computer security", "master computer security",
            "masters computer security", "information security",
        ],
    ),
    (
        "information_sciences",
        [
            "information sciences", "information science",
            "msc information sciences", "master information sciences",
            "masters information sciences",
        ],
    ),
    # ── Data & Analytics ─────────────────────────────────────────────────────
    (
        "business_analytics",
        [
            "business analytics",
            "ms ba", "msc ba", "msc business analytics",
            "master business analytics", "masters business analytics",
            "master of business analytics",
        ],
    ),
    (
        "econometrics_and_operations_research",
        [
            "econometrics", "operations research", "econometrics and operations research",
            "econometrics & operations research", "eor",
            "msc econometrics", "master econometrics", "masters econometrics",
            "msc operations research",
        ],
    ),
    # ── Business & Management ─────────────────────────────────────────────────
    (
        "business_administration",
        [
            "business administration", "mba", "msc business administration",
            "master business administration", "masters business administration",
            "master of business administration",
        ],
    ),
    (
        "accounting_and_control",
        [
            "accounting and control", "accounting & control",
            "accounting and control full time", "accounting and control ft",
            "msc accounting", "master accounting", "masters accounting and control",
        ],
    ),
    (
        "finance",
        [
            "finance", "msc finance", "master finance", "masters finance",
            "master of finance", "financial economics",
        ],
    ),
    (
        "economics",
        [
            "economics", "msc economics", "master economics", "masters economics",
            "master of economics",
        ],
    ),
    (
        "marketing",
        [
            "marketing", "msc marketing", "master marketing", "masters marketing",
            "master of marketing",
        ],
    ),
    (
        "management_policy_analysis_and_entr",
        [
            "management policy analysis", "mpa", "management policy analysis and entrepreneurship",
            "msc management policy", "master management policy",
            "masters management policy analysis",
        ],
    ),
    (
        "culture_organization_and_management",
        [
            "culture organization and management", "culture organisation and management",
            "com", "msc culture organization and management",
            "master culture organization and management",
        ],
    ),
    (
        "digital_business_and_innovation",
        [
            "digital business and innovation", "dbi",
            "msc digital business", "master digital business",
            "masters digital business and innovation",
        ],
    ),
    (
        "entrepreneurship",
        [
            "entrepreneurship", "msc entrepreneurship",
            "master entrepreneurship", "masters entrepreneurship",
            "entrepreneurship joint degree",
        ],
    ),
    (
        "transport_and_supply_chain_management",
        [
            "transport and supply chain management", "supply chain", "supply chain management",
            "tscm", "logistics",
            "msc transport and supply chain", "master supply chain management",
        ],
    ),
    # ── Law ───────────────────────────────────────────────────────────────────
    (
        "international_business_law",
        [
            "international business law", "ibl",
            "msc international business law", "master international business law",
            "masters international business law", "llm international business law",
        ],
    ),
    (
        "law_and_politics_of_international_security",
        [
            "law and politics of international security", "lpis",
            "international security law", "law international security",
            "master law and politics of international security",
            "msc law and politics of international security",
        ],
    ),
    (
        "international_crimes_conflict_and_criminology",
        [
            "international crimes conflict and criminology",
            "international crimes", "criminology", "icc",
            "master international crimes", "msc criminology",
            "masters criminology", "master criminology",
            "international crimes conflict criminology",
        ],
    ),
    # ── Social Sciences ───────────────────────────────────────────────────────
    (
        "communication_science",
        [
            "communication science", "msc communication science",
            "master communication science", "masters communication science",
            "communication studies",
        ],
    ),
    (
        "political_science",
        [
            "political science", "msc political science",
            "master political science", "masters political science",
            "politics",
        ],
    ),
    (
        "public_administration",
        [
            "public administration", "msc public administration",
            "master public administration", "masters public administration",
        ],
    ),
    (
        "social_and_cultural_anthropology",
        [
            "social and cultural anthropology", "anthropology",
            "msc anthropology", "master anthropology", "masters anthropology",
            "social anthropology", "cultural anthropology",
        ],
    ),
    (
        "social_sciences_for_a_digital_society_research",
        [
            "social sciences for a digital society", "ssds",
            "digital society research", "social sciences digital society",
            "msc social sciences digital society",
            "master social sciences for a digital society",
        ],
    ),
    (
        "spatial_transport_and_environmental_economics",
        [
            "spatial transport and environmental economics", "stee",
            "spatial economics", "environmental economics",
            "msc spatial transport", "master spatial transport and environmental economics",
        ],
    ),
    (
        "science_business_and_innovation",
        [
            "science business and innovation", "sbi",
            "msc science business and innovation",
            "master science business and innovation",
            "science and innovation",
        ],
    ),
    # ── Psychology ────────────────────────────────────────────────────────────
    (
        "clinical_and_developmental_psychopathology",
        [
            "clinical and developmental psychopathology", "cdp",
            "clinical psychopathology", "developmental psychopathology",
            "msc clinical and developmental psychopathology",
            "master clinical and developmental psychopathology",
            "masters psychopathology",
        ],
    ),
    (
        "rm_cognitive_neuropsychology",
        [
            "cognitive neuropsychology", "rm cognitive neuropsychology",
            "research master cognitive neuropsychology",
            "msc cognitive neuropsychology", "master cognitive neuropsychology",
            "neuropsychology",
        ],
    ),
    (
        "rm_social_psychology",
        [
            "social psychology", "rm social psychology",
            "research master social psychology",
            "msc social psychology", "master social psychology",
        ],
    ),
    # ── Life Sciences & Health ────────────────────────────────────────────────
    (
        "biomedical_sciences",
        [
            "biomedical sciences", "biomedical science",
            "msc biomedical sciences", "master biomedical sciences",
            "masters biomedical sciences",
        ],
    ),
    (
        "biomedical_technology_and_physics",
        [
            "biomedical technology and physics", "biomedical technology",
            "msc biomedical technology", "master biomedical technology",
            "biomedical physics",
        ],
    ),
    (
        "biomolecular_sciences",
        [
            "biomolecular sciences", "biomolecular science",
            "msc biomolecular sciences", "master biomolecular sciences",
        ],
    ),
    (
        "bioinformatics_and_systems_biology",
        [
            "bioinformatics and systems biology", "bioinformatics", "systems biology",
            "msc bioinformatics", "master bioinformatics", "masters bioinformatics",
            "bioinformatics systems biology joint degree",
        ],
    ),
    (
        "drug_discovery_sciences",
        [
            "drug discovery sciences", "drug discovery",
            "msc drug discovery", "master drug discovery sciences",
            "masters drug discovery",
        ],
    ),
    (
        "genes_in_behaviour_and_health",
        [
            "genes in behaviour and health", "genes behavior and health",
            "genes in behavior and health", "gbh",
            "msc genes in behaviour and health",
            "master genes in behaviour and health",
        ],
    ),
    (
        "health_sciences",
        [
            "health sciences", "msc health sciences",
            "master health sciences", "masters health sciences",
        ],
    ),
    (
        "global_health_research",
        [
            "global health research", "global health",
            "msc global health", "master global health",
            "masters global health research",
        ],
    ),
    (
        "neurosciences_research",
        [
            "neurosciences research", "neuroscience", "neurosciences",
            "msc neurosciences", "master neurosciences", "masters neurosciences",
            "research master neurosciences",
        ],
    ),
    (
        "oncology",
        [
            "oncology", "msc oncology", "master oncology", "masters oncology",
            "cancer research",
        ],
    ),
    (
        "personalized_medicine",
        [
            "personalized medicine", "personalised medicine",
            "msc personalized medicine", "master personalized medicine",
            "precision medicine",
        ],
    ),
    (
        "human_movement_sciences",
        [
            "human movement sciences", "human movement science",
            "msc human movement sciences", "master human movement sciences",
            "movement sciences", "sport sciences",
        ],
    ),
    (
        "human_movement_sciences",
        [
            "human movement sciences research master",
            "research master human movement sciences",
            "human movement sciences research", "rm human movement sciences",
        ],
    ),
    (
        "m_musculoskeletal_physiotherapy",
        [
            "musculoskeletal physiotherapy", "msc musculoskeletal physiotherapy",
            "master musculoskeletal physiotherapy",
            "manual physiotherapy", "m musculoskeletal physiotherapy",
        ],
    ),
    # ── Natural Sciences ──────────────────────────────────────────────────────
    (
        "mathematics",
        [
            "mathematics", "msc mathematics", "master mathematics",
            "masters mathematics", "math", "maths",
        ],
    ),
    (
        "physics_and_astronomy",
        [
            "physics and astronomy", "physics", "astronomy",
            "msc physics", "master physics", "msc physics and astronomy",
            "master physics and astronomy", "physics astronomy joint degree",
        ],
    ),
    (
        "chemistry",
        [
            "chemistry", "msc chemistry", "master chemistry",
            "masters chemistry", "chemistry joint degree",
        ],
    ),
    (
        "earth_sciences",
        [
            "earth sciences", "earth science",
            "msc earth sciences", "master earth sciences",
            "masters earth sciences", "geology", "geosciences",
        ],
    ),
    (
        "ecology_and_evolution",
        [
            "ecology and evolution", "ecology", "evolution",
            "msc ecology and evolution", "master ecology and evolution",
            "masters ecology",
        ],
    ),
    (
        "environment_and_resource_management",
        [
            "environment and resource management", "environmental management",
            "resource management", "erm",
            "msc environment and resource management",
            "master environment and resource management",
        ],
    ),
    (
        "hydrology",
        [
            "hydrology", "msc hydrology", "master hydrology",
            "masters hydrology", "water resources",
        ],
    ),
    # ── Humanities ────────────────────────────────────────────────────────────
    (
        "philosophy",
        [
            "philosophy", "msc philosophy", "master philosophy",
            "masters philosophy", "philosophy 2 years", "two year philosophy",
        ],
    ),
    (
        "humanities_research",
        [
            "humanities research", "humanities", "research master humanities",
            "msc humanities", "master humanities", "masters humanities",
        ],
    ),
    (
        "classics_and_ancient_civilizations",
        [
            "classics and ancient civilizations", "classics", "ancient civilizations",
            "msc classics", "master classics",
            "classics and ancient civilisations",
        ],
    ),
    (
        "classics_and_ancient_civilizations_research",
        [
            "classics and ancient civilizations research",
            "research master classics", "rm classics",
            "classics research master", "ancient civilizations research",
        ],
    ),
    (
        "archaeology",
        [
            "archaeology", "archeology",
            "msc archaeology", "master archaeology", "masters archaeology",
        ],
    ),
    # ── Other ─────────────────────────────────────────────────────────────────
    (
        "social_sciences_for_a_digital_society_research",
        [
            "social sciences digital society research", "ssds research",
        ],
    ),
]


class ProgramExtractor:
    """
    Lightweight rule-based extractor that identifies which master's
    program(s) a user query is targeting.
    """

    def extract(self, query: str) -> ProgramFilter:
        """
        Scan the query for known program aliases and return matched program_names.

        Args:
            query: Raw user question string.

        Returns:
            ProgramFilter with matched program_names.
            Empty list means no program detected — caller searches all programs.
        """
        query_lower = query.lower()
        result = ProgramFilter()
        seen: set[str] = set()

        for program_name, aliases in KNOWN_PROGRAMS:
            for alias in aliases:
                if alias in query_lower and program_name not in seen:
                    result.program_names.append(program_name)
                    seen.add(program_name)
                    break  # no need to check remaining aliases for this program

        return result
