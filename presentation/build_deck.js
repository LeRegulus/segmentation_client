// Génère presentation/soutenance_segmentation_bancaire.pptx — node build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");
const { applyTheme } = require("/mnt/skills/public/pptx/scripts/apply_theme.js");

const FIG = (f) => path.join(__dirname, "..", "reports", "figures", f);
const OUT = path.join(__dirname, "soutenance_segmentation_bancaire.pptx");

const THEME = {
  name: "Segmentation bancaire",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1A1F2B", lt1: "FFFFFF", dk2: "15233B", lt2: "EEF2F7",
    accent1: "7F8C8D", accent2: "2E86C1", accent3: "C0392B", accent4: "E67E22",
    accent5: "D4A017", accent6: "1F7A5C", hlink: "2E86C1", folHlink: "6D4C8D",
  },
};
const HEX = THEME.colors;
const SEGHEX = [HEX.accent1, HEX.accent2, HEX.accent3, HEX.accent4];

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.author = "Papa Alassane SECK";
pres.title = "Segmentation de clientèle bancaire par apprentissage non supervisé";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const SEG = [C.accent1, C.accent2, C.accent3, C.accent4];

// ---------------------------------------------------------------- layouts
pres.defineSlideMaster({
  title: "TITRE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 1.9, w: 11.7, h: 1.9, fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 1.0, fontSize: 20, color: C.accent5, align: "left", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENU",
  background: { color: C.background1 },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: C.text2, valign: "middle", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Segmentation de clientèle bancaire · Master BDA/IA · UNCHK", options: { x: 0.6, y: 7.05, w: 8, h: 0.3, fontSize: 10, color: C.accent1, margin: 0 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.accent1, align: "right" },
});
pres.defineSlideMaster({
  title: "SOMBRE",
  background: { color: C.text2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 32, bold: true, color: C.background1, valign: "middle", align: "left", margin: 0 }, text: "" } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontSize: 10, color: C.background2, align: "right" },
});

// ---------------------------------------------------------------- helpers
const shadow = () => ({ type: "outer", color: "000000", opacity: 0.12, blur: 6, offset: 2, angle: 90 });
function card(slide, x, y, w, h, name, fill) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill || C.background2 }, line: { type: "none" }, shadow: shadow(), objectName: name });
}
function badge(slide, x, y, d, label, color, name, fs) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color }, line: { type: "none" }, objectName: name });
  slide.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: fs || 16, bold: true, color: C.background1, margin: 0, isTextBox: true, objectName: name + " texte" });
}
function txt(slide, text, opts) { slide.addText(text, Object.assign({ isTextBox: true, margin: 0, color: C.text1, fontSize: 16, valign: "top" }, opts)); }
function img(slide, file, x, y, w, ratio, name) { slide.addImage({ path: FIG(file), x, y, w, h: w / ratio, objectName: name }); }

// ================================================================ 1. Titre
pres.addSection({ title: "Introduction" });
let s = pres.addSlide({ masterName: "TITRE", sectionTitle: "Introduction" });
s.addText("Segmentation de clientèle bancaire par apprentissage non supervisé", { placeholder: "title" });
s.addText("Bank Marketing (UCI) · 41 176 clients · Projet final — Apprentissage non supervisé", { placeholder: "body" });
[0, 1, 2, 3].forEach((i) => badge(s, 0.8 + i * 0.75, 0.9, 0.55, `S${i + 1}`, SEG[i], `Pastille S${i + 1}`, 14));
txt(s, "Papa Alassane SECK · Master Big Data Analytics / IA · UNCHK · Octobre 2026", { x: 0.8, y: 6.3, w: 11.7, h: 0.5, color: C.background2, fontSize: 16, objectName: "Auteur" });
s.addNotes("Bonjour. Je présente un projet de segmentation de la clientèle d'une banque à partir du jeu Bank Marketing de l'UCI. L'objectif : découvrir des groupes homogènes de clients pour mieux cibler les campagnes de télémarketing, sans jamais utiliser la variable de souscription pour apprendre. Les quatre pastilles colorées sont les quatre segments que nous allons découvrir ; elles reviendront tout au long de la présentation.");

// ================================================================ 2. Mission et démarche
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Introduction" });
s.addText("Découvrir des segments sans regarder la réponse", { placeholder: "title" });
const stats = [["41 176", "clients après nettoyage"], ["13", "variables de segmentation"], ["7", "familles d'algorithmes"], ["0", "usage de y à l'entraînement"]];
stats.forEach(([v, l], i) => {
  const x = 0.6 + i * 3.08;
  card(s, x, 1.55, 2.8, 1.9, `Carte chiffre ${i + 1}`);
  txt(s, v, { x: x + 0.25, y: 1.75, w: 2.3, h: 0.95, fontSize: 48, bold: true, color: i === 3 ? C.accent3 : C.text2, fontFace: "Cambria", objectName: `Chiffre ${i + 1}` });
  txt(s, l, { x: x + 0.25, y: 2.75, w: 2.3, h: 0.55, fontSize: 15, color: C.text1, objectName: `Libellé chiffre ${i + 1}` });
});
const steps = [["J1", "Préparer", "nettoyage, encodage, PCA/t-SNE/UMAP"], ["J2", "Mesurer et partitionner", "5 distances, K-means, K-medoids, K-prototypes, GMM"], ["J3", "Hiérarchie et densité", "CAH 4 liens, DBSCAN"], ["J4", "Valider", "indices internes, externes, stabilité"], ["J5", "Interpréter", "profils, souscription, actions"]];
s.addShape(pres.shapes.LINE, { x: 1.0, y: 4.55, w: 11.0, h: 0, line: { color: C.accent1, width: 1.5, dashType: "dash" }, objectName: "Frise" });
steps.forEach(([j, t, d], i) => {
  const x = 0.6 + i * 2.48;
  badge(s, x + 0.0, 4.15, 0.8, j, C.text2, `Étape ${j}`, 16);
  txt(s, t, { x, y: 5.1, w: 2.3, h: 0.45, fontSize: 16, bold: true, color: C.text2, objectName: `Titre étape ${j}` });
  txt(s, d, { x, y: 5.55, w: 2.3, h: 1.1, fontSize: 14, color: C.text1, objectName: `Détail étape ${j}` });
});
s.addNotes("La mission tient en quatre chiffres : 41 176 clients après suppression de 12 doublons, 13 variables retenues, 7 familles d'algorithmes comparées, et zéro utilisation de la variable cible pendant l'apprentissage — y ne sert qu'à évaluer à la fin. La démarche suit les cinq jours du projet : préparation, distances et partitionnement, méthodes hiérarchiques et par densité, validation, puis interprétation métier.");

// ================================================================ 3. Préparation
pres.addSection({ title: "Préparation et distances" });
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Préparation et distances" });
s.addText("Préparer : quatre décisions qui changent tout", { placeholder: "title" });
const dec = [
  ["duration exclue", "Connue seulement après l'appel : l'utiliser serait une fuite d'information."],
  ["3 indicateurs macro retirés", "emp.var.rate, nr.employed, euribor3m corrélés à 0,91–0,97 : on garde euribor3m et la confiance des ménages."],
  ["Unknown traités, pas supprimés", "Mode pour job/marital, mode par métier pour education ; default « unknown » (21 %) devient un signal."],
  ["Deux représentations", "(a) one-hot + z-score pour les méthodes euclidiennes ; (b) mixte brute pour Gower et K-prototypes."],
];
dec.forEach(([h, d], i) => {
  const y = 1.5 + i * 1.32;
  badge(s, 0.6, y + 0.05, 0.6, String(i + 1), C.text2, `Décision ${i + 1}`, 18);
  txt(s, h, { x: 1.45, y, w: 5.6, h: 0.4, fontSize: 18, bold: true, color: C.text2, objectName: `Titre décision ${i + 1}` });
  txt(s, d, { x: 1.45, y: y + 0.42, w: 5.6, h: 0.8, fontSize: 14, objectName: `Texte décision ${i + 1}` });
});
img(s, "j1_heatmap_corr.png", 7.5, 1.45, 5.25, 1185 / 1050, "Heatmap corrélations");
txt(s, "Corrélations de Pearson des variables numériques brutes", { x: 7.5, y: 6.15, w: 5.25, h: 0.35, fontSize: 11, color: C.accent1, objectName: "Légende heatmap" });
s.addNotes("Quatre décisions de préparation. Un : la durée d'appel est exclue, car on ne la connaît qu'après l'appel — c'est une fuite d'information classique sur ce jeu. Deux : les indicateurs macro-économiques sont quasi colinéaires, comme le montre la heatmap ; on n'en garde que deux pour ne pas compter quatre fois la même information. Trois : les valeurs unknown sont imputées intelligemment, et pour default, unknown devient une information en soi. Quatre : deux représentations, l'une encodée et standardisée, l'autre mixte pour la distance de Gower. Nous travaillons sur un échantillon stratifié de 4 000 clients pour les méthodes à matrice de distances.");

// ================================================================ 4. Standardisation
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Préparation et distances" });
s.addText("Sans standardisation, l'âge pèse 80 % de la distance", { placeholder: "title" });
const vars = ["age", "campaign_log", "previous", "euribor3m", "cons.conf.idx", "education_level"];
s.addChart(pres.charts.BAR, [
  { name: "Sans standardisation", labels: vars, values: [80.1, 0.2, 0.2, 2.2, 15.5, 1.9] },
  { name: "Avec standardisation", labels: vars, values: [16.9, 15.9, 18.0, 16.4, 16.5, 16.3] },
], {
  x: 0.6, y: 1.45, w: 7.6, h: 5.3, barDir: "bar", barGrouping: "clustered",
  chartColors: [HEX.accent3, HEX.accent2], showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: '0.0" %"', dataLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelColor: HEX.dk1,
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valGridLine: { style: "none" }, catGridLine: { style: "none" }, valAxisMaxVal: 95,
  showTitle: true, title: "Part de chaque variable dans la distance euclidienne² (%)", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  objectName: "Graphique contributions",
});
card(s, 8.6, 1.55, 4.1, 2.35, "Carte 32 %");
txt(s, "32 %", { x: 8.9, y: 1.7, w: 3.5, h: 1.0, fontSize: 54, bold: true, color: C.accent3, fontFace: "Cambria", objectName: "Chiffre voisins" });
txt(s, "des 10 plus proches voisins conservés quand on retire la standardisation", { x: 8.9, y: 2.75, w: 3.5, h: 1.0, fontSize: 15, objectName: "Texte voisins" });
card(s, 8.6, 4.2, 4.1, 2.45, "Carte Gower");
txt(s, "Gower ≈ Manhattan", { x: 8.9, y: 4.35, w: 3.5, h: 0.5, fontSize: 20, bold: true, color: C.text2, objectName: "Titre Gower" });
txt(s, "ρ = 0,86 avec Manhattan, 0,71 avec l'euclidienne : normalisation par l'étendue, chaque variable pèse 1/13.", { x: 8.9, y: 4.9, w: 3.5, h: 1.6, fontSize: 15, objectName: "Texte Gower" });
s.addNotes("Le chapitre 2 en une image : sans standardisation, l'âge, dont l'écart-type est d'environ dix ans, représente 80 % de la distance euclidienne au carré, et l'Euribor ou le nombre de contacts ne pèsent presque rien. Seuls 32 % des plus proches voisins survivent. Avec le z-score, chaque variable pèse environ un sixième. La distance de Gower obtient le même effet en divisant par l'étendue ; elle est très proche de Manhattan, dont elle est une moyenne normalisée. Nous avons aussi calculé Hamming et Jaccard sur le bloc binaire.");

// ================================================================ 5. Partitionnement J2
pres.addSection({ title: "Clustering" });
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Clustering" });
s.addText("Partitionner : quatre méthodes, quatre géométries", { placeholder: "title" });
const H = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 13, align: "center" } });
const cell = (t, o) => ({ text: String(t), options: Object.assign({ fontSize: 13, align: "center", color: HEX.dk1 }, o || {}) });
s.addTable([
  [H("Méthode"), H("Représentation / distance"), H("k"), H("Critère de k"), H("Silhouette (a)"), H("Davies-Bouldin"), H("Silhouette native")],
  [cell("K-means", { bold: true, align: "left" }), cell("(a) euclidienne"), cell(5), cell("coude de l'inertie"), cell("0,142"), cell("1,85"), cell("0,142")],
  [cell("K-medoids (PAM)", { bold: true, align: "left" }), cell("(b) Gower"), cell(3), cell("silhouette Gower max"), cell("0,059"), cell("2,89"), cell("0,158", { bold: true })],
  [cell("K-prototypes", { bold: true, align: "left" }), cell("(b) euclid. + γ·désaccord"), cell(5), cell("coude du coût"), cell("0,139"), cell("1,88"), cell("0,102")],
  [cell("GMM (full)", { bold: true, align: "left" }), cell("PCA 80 % (7 comp.)"), cell(6), cell("coude du BIC"), cell("0,047"), cell("2,79"), cell("—")],
], { x: 0.6, y: 1.45, w: 12.1, colW: [2.0, 2.5, 0.6, 2.2, 1.6, 1.6, 1.6], rowH: 0.42, border: { type: "solid", pt: 0.5, color: "D5DCE6" }, fill: { color: HEX.lt1 }, objectName: "Tableau comparatif J2" });
img(s, "j2_partitions_umap.png", 0.6, 3.85, 12.1, 2976 / 708, "Partitions sur UMAP");
s.addNotes("Jour 2. K-means sur la représentation encodée, K-medoids sur Gower, K-prototypes sur les données mixtes, et un modèle de mélange gaussien dans l'espace PCA. Chaque méthode choisit k selon son critère propre. Pour comparer équitablement, les indices sont recalculés dans un espace commun. K-means et K-prototypes dominent ; K-medoids a la meilleure silhouette dans sa propre géométrie mais une mauvaise vue depuis l'espace encodé : le choix de la distance change le résultat. Pour le GMM, le BIC ne cesse de baisser jusqu'à k = 10, signe que l'hypothèse gaussienne est violée par les variables discrètes.");

// ================================================================ 6. CAH et DBSCAN
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Clustering" });
s.addText("Ward pour segmenter, DBSCAN pour explorer", { placeholder: "title" });
img(s, "j3_ward_coupure.png", 0.6, 1.45, 5.9, 1720 / 595, "Dendrogramme Ward");
img(s, "j3_dbscan_umap.png", 6.8, 1.45, 5.9, 2226 / 800, "DBSCAN UMAP");
const cah = [["Single : effet de chaîne", "un cluster de 99,6–100 % des clients"], ["Average / complete", "fidèles aux distances (cophénétique 0,78), mais un groupe à 95 %"], ["Ward retenu, k = 4", "groupes équilibrés, silhouette 0,189"]];
const dbs = [["ε lu au coude des k-distances", "e* = 50 → ε* = 0,52 sur UMAP"], ["6 clusters, 11 points de bruit", "3 579 points cœur, 410 de bordure"], ["Très sensible à ε", "ε × 0,75 → 20 clusters, 314 bruits"]];
[[cah, 0.6, "CAH"], [dbs, 6.8, "DBSCAN"]].forEach(([arr, x, nm]) => arr.forEach(([h, d], i) => {
  const y = 3.75 + i * 1.02;
  txt(s, [{ text: h, options: { bold: true, color: C.text2, breakLine: true } }, { text: d, options: { color: C.text1 } }], { x, y, w: 5.9, h: 0.9, fontSize: 15, objectName: `${nm} point ${i + 1}` });
}));
s.addNotes("Jour 3. La CAH avec quatre liens : le lien simple produit l'effet de chaîne, un seul groupe géant ; average et complete sont fidèles aux distances mais isolent surtout quelques atypiques. Ward donne des groupes équilibrés ; la silhouette au-delà de k = 2 culmine à 4. DBSCAN, réglé par le graphique des k-distances, trouve 6 îlots denses sur UMAP et 11 points de bruit, mais il est très sensible à epsilon et ne sait pas affecter un nouveau client. Il complète K-means sur ses deux faiblesses : formes non convexes et points isolés.");

// ================================================================ 7. Validation
pres.addSection({ title: "Validation" });
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Validation" });
s.addText("La classification mixte Ward + K-means l'emporte", { placeholder: "title" });
const rows = [
  ["CAH Ward", 4, "0,189", "1,76", "0,153", "0,066"], ["K-means", 6, "0,151", "1,77", "0,034", "0,041"],
  ["K-prototypes", 5, "0,139", "1,88", "0,037", "0,051"], ["GMM", 4, "0,099", "2,53", "0,049", "0,045"],
  ["K-medoids (Gower)", 3, "0,059", "2,89", "0,011", "0,021"], ["DBSCAN (UMAP)", 6, "0,104", "2,25", "0,072", "0,051"],
  ["DBSCAN (PCA)", 3, "0,121", "2,22", "0,160", "0,055"],
];
const tbl = [[H("Méthode"), H("k"), H("Silhouette ↑"), H("Davies-Bouldin ↓"), H("ARI vs y"), H("V-mesure")]];
rows.forEach((r) => tbl.push(r.map((v, j) => cell(v, j === 0 ? { align: "left" } : {}))));
tbl.push(["Ward + K-means (retenue)", 4, "0,192", "1,69", "0,151", "0,062"].map((v, j) => cell(v, { bold: true, color: HEX.lt1, fill: { color: HEX.accent3 }, align: j === 0 ? "left" : "center" })));
s.addTable(tbl, { x: 0.6, y: 1.45, w: 7.9, colW: [2.6, 0.6, 1.2, 1.4, 1.0, 1.1], rowH: 0.48, border: { type: "solid", pt: 0.5, color: "D5DCE6" }, fill: { color: HEX.lt1 }, objectName: "Tableau synthèse indices" });
card(s, 8.9, 1.45, 3.8, 5.25, "Carte solution retenue");
txt(s, "Pourquoi cette solution ?", { x: 9.2, y: 1.65, w: 3.2, h: 0.5, fontSize: 18, bold: true, color: C.text2, objectName: "Titre pourquoi" });
txt(s, [
  { text: "1ʳᵉ sur silhouette, Davies-Bouldin et V-mesure parmi les solutions de 3 à 6 segments", options: { bullet: true, breakLine: true } },
  { text: "Ward fixe k et les centres ; K-means les consolide sur 41 176 clients", options: { bullet: true, breakLine: true } },
  { text: "Déployable : un nouveau client va au centre le plus proche", options: { bullet: true, breakLine: true } },
  { text: "ARI faible avec y : on décrit des profils, on ne reproduit pas la souscription", options: { bullet: true } },
], { x: 9.2, y: 2.25, w: 3.2, h: 4.3, fontSize: 14, paraSpaceAfter: 8, objectName: "Arguments solution" });
s.addNotes("Jour 4. Nous avons calculé silhouette, Davies-Bouldin et Dunn pour toutes les méthodes et toutes les valeurs de k, ainsi que les indices externes. Parmi les solutions actionnables, de 3 à 6 segments sans segment de moins de 5 %, la CAH Ward à k = 4 est première sur les trois critères. Comme la CAH ne passe pas à l'échelle et n'affecte pas de nouveaux clients, on la consolide par K-means initialisé avec ses barycentres : c'est la classification mixte classique. Elle améliore encore silhouette et Davies-Bouldin. Un ARI faible avec y est normal : une segmentation n'a pas vocation à prédire la souscription.");

// ================================================================ 8. Stabilité
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Validation" });
s.addText("Stabilité : la partition finale résiste au rééchantillonnage", { placeholder: "title" });
const ks = ["2", "3", "4", "5", "6", "7", "8", "9", "10"];
s.addChart(pres.charts.LINE, [
  { name: "Bootstrap (20 réplications)", labels: ks, values: [0.989, 0.717, 0.945, 0.971, 0.902, 0.903, 0.887, 0.762, 0.786] },
  { name: "20 graines, n_init = 1", labels: ks, values: [0.932, 0.605, 0.666, 0.86, 0.73, 0.649, 0.692, 0.672, 0.662] },
], {
  x: 0.6, y: 1.45, w: 7.6, h: 5.3, chartColors: [HEX.accent2, HEX.accent4], lineSize: 2.5, lineDataSymbolSize: 8,
  showLegend: true, legendPos: "b", legendFontSize: 12, legendFontFace: "+mn-lt",
  valAxisMinVal: 0.5, valAxisMaxVal: 1, valAxisLabelFormatCode: "0.0", valAxisLabelColor: HEX.dk1, valAxisLabelFontSize: 12, valAxisLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 12, catAxisLabelFontFace: "+mn-lt", catAxisTitle: "k", showCatAxisTitle: true, catAxisTitleFontSize: 12,
  valGridLine: { color: "E3E8EF", size: 0.5 }, catGridLine: { style: "none" },
  showTitle: true, title: "ARI moyen entre exécutions de K-means", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  objectName: "Graphique stabilité",
});
card(s, 8.6, 1.55, 4.1, 2.3, "Carte 0,999");
txt(s, "0,999", { x: 8.9, y: 1.7, w: 3.5, h: 1.0, fontSize: 54, bold: true, color: C.accent6, fontFace: "Cambria", objectName: "Chiffre stabilité" });
txt(s, "ARI bootstrap de la solution retenue, centres Ward fixés", { x: 8.9, y: 2.75, w: 3.5, h: 0.9, fontSize: 15, objectName: "Texte stabilité" });
card(s, 8.6, 4.15, 4.1, 2.5, "Carte limite");
txt(s, "Limite assumée", { x: 8.9, y: 4.3, w: 3.5, h: 0.5, fontSize: 20, bold: true, color: C.accent3, objectName: "Titre limite" });
txt(s, "Chaîne complète CAH → K-means : ARI 0,72 ± 0,23. Figer les centres et les ré-estimer périodiquement.", { x: 8.9, y: 4.85, w: 3.5, h: 1.6, fontSize: 15, objectName: "Texte limite" });
s.addNotes("La stabilité. En relançant K-means avec 20 graines et une seule initialisation, l'accord moyen varie de 0,6 à 0,9 : d'où l'usage de plusieurs initialisations. En bootstrap, k = 2, 4 et 5 sont stables, k = 3 ne l'est pas. Une fois les centres Ward fixés, notre partition finale est quasiment invariante : ARI de 0,999. En revanche, si l'on refait toute la chaîne, CAH comprise, sur chaque bootstrap, l'ARI tombe à 0,72 : la CAH coupe parfois différemment les petits groupes. Recommandation : figer les centres de référence et les recalculer périodiquement.");

// ================================================================ 9. Les 4 segments
pres.addSection({ title: "Segments et actions" });
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Segments et actions" });
s.addText("Quatre segments aux profils contrastés", { placeholder: "title" });
const segs = [
  ["Masse froide", "65 %", ["Jamais contactés", "51 % sur ligne fixe", "Taux élevés (Euribor 4,8 %)", "2,8 appels en moyenne"]],
  ["Jeunes actifs mobiles", "17 %", ["38 ans, 36 % célibataires", "90 % sur mobile", "Taux bas (Euribor 1,3 %)", "Ménages inquiets (−46)"]],
  ["Seniors épargnants", "6 %", ["47 ans, 21 % retraités", "3 % défaut non déclaré", "37 % déjà contactés", "Taux au plancher (0,8 %)"]],
  ["Déjà sollicités", "11 %", ["Tous contactés avant", "80 % échecs, 20 % succès", "93 % sur mobile", "Taux bas (Euribor 1,6 %)"]],
];
segs.forEach(([n, p, lines], i) => {
  const x = 0.6 + i * 3.08;
  card(s, x, 1.5, 2.85, 5.2, `Carte segment ${i + 1}`, C.background1);
  badge(s, x + 0.25, 1.75, 0.8, `S${i + 1}`, SEG[i], `Pastille segment ${i + 1}`, 18);
  txt(s, p, { x: x + 1.2, y: 1.75, w: 1.5, h: 0.8, fontSize: 36, bold: true, color: SEG[i], fontFace: "Cambria", valign: "middle", objectName: `Part segment ${i + 1}` });
  txt(s, n, { x: x + 0.25, y: 2.75, w: 2.4, h: 0.85, fontSize: 18, bold: true, color: C.text2, objectName: `Nom segment ${i + 1}` });
  txt(s, lines.map((l, j) => ({ text: l, options: { bullet: true, breakLine: j < lines.length - 1 } })), { x: x + 0.25, y: 3.65, w: 2.45, h: 2.9, fontSize: 14, paraSpaceAfter: 6, objectName: `Profil segment ${i + 1}` });
});
s.addNotes("Voici les quatre segments sur la population complète. S1, la masse froide : 65 % des clients, jamais contactés, joints pour moitié sur fixe, en période de taux élevés, et la plus forte pression d'appel. S2, les jeunes actifs mobiles : plus jeunes, plus de célibataires, sur mobile, en période de taux bas. S3, les seniors épargnants : 47 ans en moyenne, beaucoup de retraités, presque aucun défaut non déclaré. S4, les clients déjà sollicités : tous contactés lors d'une campagne précédente, majoritairement sans succès.");

// ================================================================ 10. Taux de souscription
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Segments et actions" });
s.addText("35 % des clients concentrent 72 % des souscripteurs", { placeholder: "title" });
const segLab = ["S1 Masse froide", "S2 Jeunes actifs", "S3 Seniors", "S4 Déjà sollicités"];
const rates = [4.9, 17.3, 42.5, 22.0];
s.addChart(pres.charts.BAR, rates.map((v, i) => ({ name: segLab[i], labels: segLab, values: rates.map((_, j) => (j === i ? v : 0)) })), {
  x: 0.6, y: 1.45, w: 7.6, h: 5.3, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
  chartColors: SEGHEX, showLegend: false,
  showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: '0.0" %";;;', dataLabelFontSize: 14, dataLabelFontBold: true, dataLabelColor: HEX.lt1, dataLabelFontFace: "+mn-lt",
  catAxisLabelColor: HEX.dk1, catAxisLabelFontSize: 14, catAxisLabelFontFace: "+mn-lt", catAxisOrientation: "maxMin",
  valAxisHidden: true, valAxisMaxVal: 45, valGridLine: { style: "none" }, catGridLine: { style: "none" },
  showTitle: true, title: "Taux de souscription au dépôt à terme (moyenne : 11,3 %)", titleFontSize: 14, titleColor: HEX.dk2, titleFontFace: "+mn-lt",
  objectName: "Graphique taux de souscription",
});
card(s, 8.6, 1.55, 4.1, 2.3, "Carte lift");
txt(s, "× 3,8", { x: 8.9, y: 1.7, w: 3.5, h: 1.0, fontSize: 54, bold: true, color: C.accent3, fontFace: "Cambria", objectName: "Chiffre lift" });
txt(s, "lift des seniors épargnants : 42,5 % de souscription contre 11,3 %", { x: 8.9, y: 2.75, w: 3.5, h: 0.9, fontSize: 15, objectName: "Texte lift" });
card(s, 8.6, 4.15, 4.1, 2.5, "Carte période");
txt(s, "Effet de période", { x: 8.9, y: 4.3, w: 3.5, h: 0.5, fontSize: 20, bold: true, color: C.text2, objectName: "Titre période" });
txt(s, "S1 = appels de 2008 (taux élevés) ; S3 n'apparaît qu'en 2010. Le contexte de taux compte autant que le client.", { x: 8.9, y: 4.85, w: 3.5, h: 1.6, fontSize: 15, objectName: "Texte période" });
s.addNotes("Ce n'est qu'ici que la cible intervient, a posteriori. Les taux de souscription vont de 4,9 % pour la masse froide à 42,5 % pour les seniors épargnants, soit un lift de 3,8. Les segments S2, S3 et S4 ne représentent que 35 % des clients mais 72 % des souscripteurs. Point d'honnêteté : l'analyse chronologique montre un fort effet de période — S1 correspond aux appels de 2008, quand les taux étaient élevés. Une partie de la segmentation reflète donc le contexte économique, ce qui reste actionnable : savoir quand appeler.");

// ================================================================ 11. Recommandations
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Segments et actions" });
s.addText("Une action marketing par segment", { placeholder: "title" });
const recos = [
  ["Réduire la pression", "Plafonner à 2 appels ; SMS, e-mail, application ; appel conseiller seulement après un signal d'intérêt.", "Canal bas coût"],
  ["Mobile-first", "SMS + rappel ; dépôt à terme court et accessible ; souscription 100 % digitale.", "Digital"],
  ["Conseiller dédié", "Dépôt long à taux garanti, sécurité du capital, rendez-vous en agence ; un appel suffit.", "Priorité 1"],
  ["Exploiter l'historique", "Anciens souscripteurs : offre de renouvellement. Échecs : nouvelle offre, relances espacées.", "Fidélisation"],
];
recos.forEach(([h, d, tag], i) => {
  const y = 1.5 + i * 1.3;
  card(s, 0.6, y, 12.1, 1.1, `Ligne reco ${i + 1}`, C.background2);
  badge(s, 0.85, y + 0.15, 0.8, `S${i + 1}`, SEG[i], `Pastille reco ${i + 1}`, 18);
  txt(s, h, { x: 1.95, y: y + 0.12, w: 2.9, h: 0.85, fontSize: 18, bold: true, color: C.text2, valign: "middle", objectName: `Titre reco ${i + 1}` });
  txt(s, d, { x: 4.9, y: y + 0.12, w: 5.9, h: 0.85, fontSize: 15, valign: "middle", objectName: `Texte reco ${i + 1}` });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 10.95, y: y + 0.3, w: 1.55, h: 0.5, rectRadius: 0.25, fill: { color: SEG[i] }, line: { type: "none" }, objectName: `Étiquette reco ${i + 1}` });
  txt(s, tag, { x: 10.95, y: y + 0.3, w: 1.55, h: 0.5, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `Texte étiquette ${i + 1}` });
});
s.addNotes("Une action par segment. S1 : arrêter de sur-solliciter, basculer vers des canaux à bas coût et n'appeler qu'après un signal d'intérêt. S2 : une campagne mobile-first avec un produit court et accessible, souscriptible en ligne. S3 : la priorité absolue, confiée à des conseillers dédiés, avec un produit long et sécurisé ; inutile d'insister, un appel suffit. S4 : exploiter l'historique, en relançant les anciens souscripteurs avec une offre de renouvellement et en changeant d'argumentaire pour les échecs.");

// ================================================================ 12. Bonus
pres.addSection({ title: "Extensions" });
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Extensions" });
s.addText("Pour aller plus loin : quatre extensions", { placeholder: "title" });
const bonus = [
  ["Anomalies", "54 %", "de souscription chez les 411 atypiques d'Isolation Forest : des niches à forte valeur, pas des erreurs.", C.accent3],
  ["Sélection de variables", "1,0", "de silhouette avec housing + loan seuls : le biais du wrapper. Sans macro, la dépendance à la période passe de 0,54 à 0,31.", C.accent2],
  ["Application Streamlit", "100 %", "d'accord avec le modèle sur 300 clients ; saisie individuelle ou affectation d'un fichier CSV.", C.accent6],
  ["Online Retail (RFM)", "65 %", "du chiffre d'affaires réalisé par les Champions, 16 % des clients ; silhouette 0,34.", C.accent4],
];
bonus.forEach(([h, v, d, col], i) => {
  const x = 0.6 + (i % 2) * 6.15, y = 1.5 + Math.floor(i / 2) * 2.65;
  card(s, x, y, 5.95, 2.4, `Carte bonus ${i + 1}`);
  txt(s, h, { x: x + 0.3, y: y + 0.2, w: 5.3, h: 0.45, fontSize: 18, bold: true, color: C.text2, objectName: `Titre bonus ${i + 1}` });
  txt(s, v, { x: x + 0.3, y: y + 0.75, w: 1.9, h: 1.3, fontSize: 40, bold: true, color: col, fontFace: "Cambria", valign: "middle", objectName: `Chiffre bonus ${i + 1}` });
  txt(s, d, { x: x + 2.25, y: y + 0.75, w: 3.4, h: 1.45, fontSize: 14, valign: "middle", objectName: `Texte bonus ${i + 1}` });
});
s.addNotes("Les quatre extensions bonus. Isolation Forest isole 1 % de clients atypiques : plus âgés, avec un long historique ; ils souscrivent à 54 %. La sélection de variables montre le piège du wrapper non supervisé : deux variables binaires donnent une silhouette parfaite sans aucun sens métier ; retirer les indicateurs macro rend la segmentation moins dépendante de la période. L'application Streamlit affecte un nouveau client. Enfin, la démarche a été reproduite sur Online Retail en RFM : les Champions, 16 % des clients, font 65 % du chiffre d'affaires.");

// ================================================================ 13. Démo Streamlit
s = pres.addSlide({ masterName: "CONTENU", sectionTitle: "Extensions" });
s.addText("Démo : affecter un nouveau client en un clic", { placeholder: "title" });
s.addImage({ path: FIG("app_streamlit_crop.png"), x: 0.6, y: 1.4, w: 6.5, h: 6.5 * 920 / 1120, objectName: "Capture application" });
const demo = [["Saisir le profil", "âge, métier, prêts, canal, historique, contexte de taux"], ["Même pipeline qu'à l'entraînement", "transform_new_client() puis centre le plus proche"], ["Lire la fiche", "segment, taux historique, lift, action recommandée"], ["Traiter un lot", "CSV au format UCI, export des affectations"]];
demo.forEach(([h, d], i) => {
  const y = 1.55 + i * 1.3;
  badge(s, 7.5, y, 0.6, String(i + 1), C.accent6, `Étape démo ${i + 1}`, 16);
  txt(s, [{ text: h, options: { bold: true, color: C.text2, breakLine: true, fontSize: 17 } }, { text: d, options: { fontSize: 15 } }], { x: 8.3, y: y - 0.05, w: 4.4, h: 1.1, objectName: `Texte démo ${i + 1}` });
});
txt(s, "streamlit run app/streamlit_app.py", { x: 7.5, y: 6.55, w: 5.2, h: 0.35, fontSize: 13, fontFace: "Courier New", color: C.accent1, objectName: "Commande lancement" });
s.addNotes("Démonstration de l'application. Le conseiller saisit le profil du client ; l'application applique exactement le même prétraitement qu'à l'entraînement, puis affecte le client au centre le plus proche. Elle affiche le segment, le taux de souscription historique, le lift, l'action recommandée et la position du client sur une projection PCA. Un second onglet traite un fichier CSV entier. Si le temps le permet, je fais la démonstration en direct.");

// ================================================================ 14. Conclusion
pres.addSection({ title: "Conclusion" });
s = pres.addSlide({ masterName: "SOMBRE", sectionTitle: "Conclusion" });
s.addText("Ce qu'il faut retenir", { placeholder: "title" });
const conc = [
  ["La distance compte autant que l'algorithme", "Standardisation et Gower changent les voisins et les partitions."],
  ["Lire les indices ensemble", "La meilleure silhouette (lien single, k = 2) est la pire segmentation."],
  ["4 segments, de 4,9 % à 42,5 % de souscription", "Classification mixte Ward + K-means, stable et déployable."],
];
conc.forEach(([h, d], i) => {
  const y = 1.6 + i * 1.25;
  badge(s, 0.6, y, 0.7, String(i + 1), C.accent5, `Point clé ${i + 1}`, 18);
  txt(s, [{ text: h, options: { bold: true, breakLine: true, fontSize: 20, color: C.background1 } }, { text: d, options: { fontSize: 16, color: C.background2 } }], { x: 1.55, y: y - 0.05, w: 6.3, h: 1.1, objectName: `Texte point clé ${i + 1}` });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 8.3, y: 1.55, w: 4.4, h: 3.7, rectRadius: 0.12, fill: { color: C.background1, transparency: 88 }, line: { color: C.accent5, width: 1 }, objectName: "Cadre limites" });
txt(s, "Limites et suites", { x: 8.6, y: 1.75, w: 3.8, h: 0.5, fontSize: 20, bold: true, color: C.accent5, objectName: "Titre limites" });
txt(s, [
  { text: "Silhouettes modestes : groupes recouvrants", options: { bullet: true, breakLine: true } },
  { text: "Forte dépendance au contexte de taux", options: { bullet: true, breakLine: true } },
  { text: "Double segmentation : profil pur (CRM) + contexte (campagnes)", options: { bullet: true, breakLine: true } },
  { text: "Test A/B des recommandations", options: { bullet: true } },
], { x: 8.6, y: 2.35, w: 3.9, h: 2.8, fontSize: 15, color: C.background1, paraSpaceAfter: 8, objectName: "Texte limites" });
txt(s, "Merci — questions ?", { x: 0.6, y: 5.75, w: 12.1, h: 0.9, fontSize: 36, bold: true, color: C.background1, fontFace: "Cambria", objectName: "Merci" });
s.addNotes("Trois messages pour conclure. Un : la représentation et la distance pèsent autant que l'algorithme. Deux : les indices de validation se lisent ensemble et avec des contraintes métier — la meilleure silhouette correspond ici à la pire segmentation. Trois : nous livrons quatre segments interprétables, stables et déployables, aux taux de souscription très contrastés. Les limites : des silhouettes modestes et une dépendance au contexte de taux, d'où l'idée d'une double segmentation et d'un test A/B des recommandations. Merci, je suis prêt pour vos questions.");

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("OK", OUT);
})();
