import time, requests
import googlemaps
import pandas as pd

from model import POI_TYPES
 
GMAPS_KEY = ""
gmaps     = googlemaps.Client(GMAPS_KEY)
RADIUS    = 1600
 
CAMPUSES = [
    ("University of Michigan",                "University of Michigan Ann Arbor"),
    ("Ohio State University",                 "Ohio State University Columbus"),
    ("UCLA",                                  "UCLA Los Angeles"),
    ("MIT",                                   "MIT Cambridge MA"),
    ("University of Texas at Austin",         "UT Austin Texas"),
    ("Penn State University",                 "Penn State University Park PA"),
    ("University of Florida",             "ASU Tempe Arizona"),
    ("University of Wisconsin-Madison",       "UW Madison Wisconsin"),
    ("UC Berkeley",                           "UC Berkeley California"),
    ("University of Washington",              "University of Washington Seattle"),
    ("Indiana University",                    "Indiana University Bloomington"),
    ("Purdue University",                     "Purdue University West Lafayette"),
    ("University of Illinois Urbana-Champaign", "UIUC Champaign Illinois"),
    ("Michigan State University",             "Michigan State East Lansing"),
    ("University of Minnesota",               "University of Minnesota Minneapolis"),
    ("University of Colorado Boulder",        "University of Florida Gainesville"),
    ("NYU",                                   "New York University Manhattan"),
    ("Arizona State University",                  "CU Boulder Colorado"),
    ("University of Oregon",                  "University of Oregon Eugene"),
    ("University of Arizona",                 "University of Arizona Tucson"),
    ("Georgia Tech",                          "Georgia Tech Atlanta"),
    ("University of Georgia",                 "University of Georgia Athens"),
    ("Clemson University",                    "Clemson University South Carolina"),
    ("Duke University",                       "Duke University Durham NC"),
    ("University of North Carolina",          "UNC Chapel Hill"),
    ("Northwestern University",               "Northwestern University Evanston IL"),
    ("University of Chicago",                 "University of Chicago Hyde Park"),
    ("Vanderbilt University",                 "Vanderbilt University Nashville"),
    ("Tulane University",                     "Tulane University New Orleans"),
    ("University of Miami",                   "University of Miami Coral Gables"),
    ("Boston University",                     "Boston University Massachusetts"),
    ("Northeastern University",               "Northeastern University Boston"),
    ("Tufts University",                      "Tufts University Medford MA"),
    ("Georgetown University",                 "Georgetown University Washington DC"),
    ("George Washington University",          "GWU Washington DC"),
    ("American University",                   "American University Washington DC"),
    ("University of Southern California",     "USC Los Angeles"),
    ("Stanford University",                   "Stanford University Palo Alto"),
    ("UC San Diego",                          "UCSD La Jolla California"),
    ("UC Davis",                              "UC Davis California"),
    ("UC Santa Barbara",                      "UCSB Santa Barbara"),
    ("Oregon State University",               "Oregon State Corvallis"),
    ("Washington State University",           "WSU Pullman Washington"),
    ("University of Utah",                    "University of Utah Salt Lake City"),
    ("Brigham Young University",              "BYU Provo Utah"),
    ("Colorado State University",             "CSU Fort Collins Colorado"),
    ("University of New Mexico",              "UNM Albuquerque"),
    ("University of Nevada Las Vegas",        "UNLV Las Vegas"),
    ("Boise State University",                "Boise State Idaho"),
    ("University of Idaho",                   "University of Idaho Moscow"),
    ("Iowa State University",                 "Iowa State Ames Iowa"),
    ("University of Iowa",                    "University of Iowa Iowa City"),
    ("University of Nebraska",                "UNL Lincoln Nebraska"),
    ("Kansas State University",               "Kansas State Manhattan KS"),
    ("University of Kansas",                  "KU Lawrence Kansas"),
    ("University of Missouri",                "Mizzou Columbia Missouri"),
    ("Saint Louis University",                "SLU St Louis Missouri"),
    ("Washington University in St Louis",     "WashU St Louis"),
    ("University of Kentucky",                "UK Lexington Kentucky"),
    ("University of Louisville",              "UofL Louisville Kentucky"),
    ("University of Tennessee",               "UT Knoxville Tennessee"),
    ("Belmont University",                    "Belmont University Nashville"),
    ("Auburn University",                     "Auburn University Alabama"),
    ("University of Alabama",                 "University of Alabama Tuscaloosa"),
    ("Mississippi State University",          "Mississippi State Starkville"),
    ("Louisiana State University",            "LSU Baton Rouge"),
    ("University of Arkansas",                "University of Arkansas Fayetteville"),
    ("Oklahoma State University",             "Oklahoma State Stillwater"),
    ("University of Oklahoma",                "OU Norman Oklahoma"),
    ("Texas A&M University",                  "Texas A&M College Station"),
    ("Texas Tech University",                 "Texas Tech Lubbock"),
    ("Rice University",                       "Rice University Houston"),
    ("Baylor University",                     "Baylor University Waco"),
    ("TCU",                                   "TCU Fort Worth Texas"),
    ("SMU",                                   "SMU Dallas Texas"),
    ("University of Houston",                 "University of Houston Texas"),
    ("University of Virginia",                "UVA Charlottesville"),
    ("Virginia Tech",                         "Virginia Tech Blacksburg"),
    ("William and Mary",                      "College of William and Mary Williamsburg VA"),
    ("University of Maryland",                "UMD College Park Maryland"),
    ("University of Delaware",                "University of Delaware Newark DE"),
    ("Rutgers University",                    "Rutgers New Brunswick NJ"),
    ("Princeton University",                  "Princeton University New Jersey"),
    ("Yale University",                       "Yale University New Haven CT"),
    ("Harvard University",                    "Harvard University Cambridge MA"),
    ("Columbia University",                   "Columbia University New York"),
    ("Cornell University",                    "Cornell University Ithaca NY"),
    ("University of Rochester",               "University of Rochester New York"),
    ("Syracuse University",                   "Syracuse University New York"),
    ("Fordham University",                    "Fordham University Bronx NY"),
    ("University of Pittsburgh",              "University of Pittsburgh Pennsylvania"),
    ("Carnegie Mellon University",            "CMU Pittsburgh Pennsylvania"),
    ("Temple University",                     "Temple University Philadelphia"),
    ("Drexel University",                     "Drexel University Philadelphia"),
    ("Penn",                                  "University of Pennsylvania Philadelphia"),
    ("Villanova University",                  "Villanova University Pennsylvania"),
    ("Case Western Reserve University",       "Case Western Cleveland Ohio"),
    ("University of Cincinnati",              "University of Cincinnati Ohio"),
    ("Miami University",                      "Miami University Oxford Ohio"),
    ("Bowling Green State University",        "BGSU Bowling Green Ohio"),
    ("Marquette University",                  "Marquette University Milwaukee"),
    ("University of Wisconsin-Milwaukee",     "UWM Milwaukee Wisconsin"),
]
 
 
def geocode_campus(query: str):
    results = gmaps.geocode(query)
    if not results:
        return None, None
    loc = results[0]["geometry"]["location"]
    return loc["lat"], loc["lng"]
 
 
def fetch_pois(lat: float, lng: float) -> dict:
    counts = {}
    for label, ptype in POI_TYPES.items():
        try:
            r = gmaps.places_nearby(location=(lat, lng), radius=RADIUS, type=ptype)
            counts[label] = len(r.get("results", []))
            time.sleep(0.12)
        except Exception as e:
            print(f"    POI error ({label}): {e}")
            counts[label] = 0
    return counts
 
 
def fetch_weather(lat: float, lng: float) -> dict:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lng}"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,sunshine_duration"
        "&timezone=auto&forecast_days=7"
    )
    try:
        data = requests.get(url, timeout=8).json().get("daily", {})
        avg_max    = sum(data["temperature_2m_max"]) / 7
        avg_min    = sum(data["temperature_2m_min"]) / 7
        avg_precip = sum(data["precipitation_sum"]) / 7
        avg_sun    = sum(data["sunshine_duration"]) / 7
        return {
            "avg_temp_c":    round((avg_max + avg_min) / 2, 1),
            "avg_precip_mm": round(avg_precip, 2),
            "avg_sun_hrs":   round(avg_sun / 3600, 1),
        }
    except Exception as e:
        print(f"    Weather error: {e}")
        return {"avg_temp_c": None, "avg_precip_mm": None, "avg_sun_hrs": None}
 
 
def build_dataset(out_path="campus_features.csv"):
    rows = []
    for name, query in CAMPUSES:
        print(f"[{len(rows)+1}/{len(CAMPUSES)}] {name}")
        lat, lng = geocode_campus(query)
        if lat is None:
            print("  ✗ geocode failed, skipping")
            continue
        pois    = fetch_pois(lat, lng)
        weather = fetch_weather(lat, lng)
        row     = {"name": name, "lat": lat, "lng": lng, **pois, **weather}
        rows.append(row)
        print(f"  ✓ lat={lat:.4f} lng={lng:.4f}  pois={sum(pois.values())}")
        time.sleep(0.3)
 
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} campuses → {out_path}")
    return df
 
 
if __name__ == "__main__":
    build_dataset()
 