"""Fiches des 4 segments retenus (classification mixte CAH Ward + consolidation K-means, k = 4).

Les étiquettes 0..3 correspondent aux sorties de ``models/kmeans_final.joblib``.
Les chiffres cités proviennent du notebook 05 (population complète, n = 41 176).
"""

SEGMENTS = {
    0: {
        "nom": "Masse froide (fixe, taux élevés)",
        "court": "Masse froide",
        "profil": ("65 % des clients ; jamais contactés auparavant ; appelés en période de taux élevés "
                   "(Euribor ≈ 4,8 %) ; 51 % joints sur ligne fixe ; 26 % n'ont pas déclaré leur situation de "
                   "défaut ; pression d'appel la plus forte (2,8 appels en moyenne)."),
        "reco": ("Réduire la pression téléphonique : plafonner à 2 appels, basculer sur des canaux à faible "
                 "coût (SMS, e-mail, application) et ne déclencher l'appel conseiller qu'après un signal "
                 "d'intérêt (clic, simulation). Compléter les données manquantes (défaut, mobile) avant "
                 "toute relance."),
        "taux": 4.9, "lift": 0.43,
    },
    1: {
        "nom": "Jeunes actifs mobiles (taux bas)",
        "court": "Jeunes actifs mobiles",
        "profil": ("17 % des clients ; 38 ans en moyenne, 36 % de célibataires ; jamais contactés ; joignables "
                   "sur mobile (90 %) ; appelés en période de taux bas (Euribor ≈ 1,3 %) et de forte "
                   "inquiétude des ménages (confiance −46)."),
        "reco": ("Campagne mobile-first (SMS + rappel) proposant un dépôt à terme court et accessible "
                 "(petit ticket d'entrée, sortie anticipée possible), argumentée sur l'épargne de "
                 "précaution ; appels en fin de journée ou parcours 100 % digital de souscription."),
        "taux": 17.3, "lift": 1.54,
    },
    2: {
        "nom": "Seniors et retraités épargnants",
        "court": "Seniors épargnants",
        "profil": ("6 % des clients ; 47 ans en moyenne, 21 % de retraités (contre 4 % ailleurs) ; quasiment "
                   "aucun défaut non déclaré (3 %) ; 37 % déjà contactés dont près de la moitié avec succès ; "
                   "appelés au plus bas des taux (Euribor ≈ 0,8 %)."),
        "reco": ("Segment prioritaire (lift ×3,8) : affecter des conseillers dédiés, proposer des dépôts à "
                 "terme longs à taux garanti mettant en avant la sécurité du capital, avec rendez-vous en "
                 "agence ; un seul appel suffit en moyenne (1,8), inutile d'insister."),
        "taux": 42.5, "lift": 3.77,
    },
    3: {
        "nom": "Clients déjà sollicités",
        "court": "Déjà sollicités",
        "profil": ("11 % des clients ; tous contactés lors d'une campagne précédente (1,3 contact en moyenne ; "
                   "80 % d'échecs, 20 % de succès) ; 93 % sur mobile ; période de taux bas."),
        "reco": ("Exploiter l'historique : recontacter en priorité les anciens souscripteurs (offre de "
                 "renouvellement / fidélité) ; pour les échecs, changer d'offre et d'argumentaire plutôt que "
                 "répéter le même appel, espacer les relances et limiter à 2 tentatives."),
        "taux": 22.0, "lift": 1.95,
    },
}

SEG_COLORS = {0: "#7F8C8D", 1: "#2E86C1", 2: "#C0392B", 3: "#E67E22"}
