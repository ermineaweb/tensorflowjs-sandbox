exports.MSZoning = (val) => {
  // MSZoning: Identifies the general zoning classification of the sale.
  //     MSZoning: identifie la classification de zonage générale de la vente.
  //
  //     A	Agriculture
  // C	Commercial
  // FV	Floating Village Residential
  // I	Industrial
  // RH	Residential High Density
  // RL	Residential Low Density
  // RP	Residential Low Density Park
  // RM	Residential Medium Density
  switch (val) {
    case "A":
      return 1;
    case "C":
      return 2;
    case "FV":
      return 3;
    case "I":
      return 4;
    case "RH":
      return 5;
    case "RL":
      return 6;
    case "RP":
      return 7;
    case "RM":
      return 8;
    default:
      return 0;
  }
}

exports.Street = (val) => {
  // Street: Type of road access to property
  // Rue: Type de route d'accès à la propriété
  //
  // Grvl	Gravel
  // Pave	Paved
  switch (val) {
    case "Grvl":
      return 1;
    case "Pave":
      return 2;
    default:
      return 0;
  }
}
exports.Alley = (val) => {
    // Alley: Type of alley access to property
    // Ruelle: Type de ruelle d'accès à la propriété
    //
    // Grvl	Gravel
    // Pave	Paved
    // NA 	No alley access
  switch (val) {
    case "Grvl":
      return 1;
    case "Pave":
      return 2;
    default:
      return 0;
  }
}

exports.LotShape = (val) => {
    // LotShape: General shape of property
    // LotShape: Forme générale de la propriété
    //
    // Reg	Regular
    // IR1	Slightly irregular
    // IR2	Moderately Irregular
    // IR3	Irregular
  switch (val) {
    case "Reg":
      return 1;
    case "IR1":
      return 2;
    case "IR2":
      return 3;
    case "IR3":
      return 4;
    default:
      return 0;
  }
}
exports.LandContour = (val) => {
  // LandContour: Flatness of the property
  // LandContour: Planéité de la propriété
  //
  // Lvl	Near Flat/Level
  // Bnk	Banked - Quick and significant rise from street grade to building
  // HLS	Hillside - Significant slope from side to side
  // Low	Depression
  switch (val) {
    case "Lvl":
      return 1;
    case "Bnk":
      return 2;
    case "HLS":
      return 3;
    case "Low":
      return 4;
    default:
      return 0;
  }
}

exports.Utilities = (val) => {
    // Utilities: Type of utilities available
    // Utilitaires: type d'utilitaires disponibles
    //
    // AllPub	All public Utilities (E,G,W,& S)
    // NoSewr	Electricity, Gas, and Water (Septic Tank)
    // NoSeWa	Electricity and Gas Only
    // ELO	Electricity only
  switch (val) {
    case "AllPub":
      return 1;
    case "NoSewr":
      return 2;
    case "NoSeWa":
      return 3;
    case "ELO":
      return 4;
    default:
      return 0;
  }
}

exports.LotConfig = (val) => {
    // LotConfig: Lot configuration
    // LotConfig: Configuration du lot
    //
    // Inside	Inside lot
    // Corner	Corner lot
    // CulDSac	Cul-de-sac
    // FR2	Frontage on 2 sides of property
    // FR3	Frontage on 3 sides of property
  switch (val) {
    case "Inside":
      return 1;
    case "Corner":
      return 2;
    case "CulDSac":
      return 3;
    case "FR2":
      return 4;
    case "FR3":
      return 4;
    default:
      return 0;
  }
}

exports.LandSlope = (val) => {
    // LandSlope: Slope of property
    // LandSlope: Pente de la propriété
    //
    // Gtl	Gentle slope
    // Mod	Moderate Slope
    // Sev	Severe Slope
  switch (val) {
    case "Gtl":
      return 1;
    case "Mod":
      return 2;
    case "Sev":
      return 3;
    default:
      return 0;
  }
}

exports.RoofStyle = (val) => {
  // RoofStyle: Type of roof
  // RoofStyle: Type de toit
  //
  // Flat	Flat
  // Gable	Gable
  // Gambrel	Gabrel (Barn)
  // Hip	Hip
  // Mansard	Mansard
  // Shed	Shed
  switch (val) {
    case "Flat":
      return 1;
    case "Gable":
      return 2;
    case "Gambrel":
      return 3;
    case "Hip":
      return 3;
    case "Mansard":
      return 3;
    case "Shed":
      return 3;
    default:
      return 0;
  }
}
