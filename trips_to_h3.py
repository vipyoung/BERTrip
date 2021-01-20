"""
Parse trips.jsonl to generate a file in which
each row is a trip represented as a sequence of h3 cells
of a given resolution
"""

import h3
import json
from datetime import datetime

if __name__ == '__main__':
    # chech resolution table here: https://h3geo.org/docs/core-library/restable
    resolution = 10
    #with open('data/trips.jsonl') as f, \
    with open('data/trips_pred.jsonl') as f, \
        open('data/processed/trips_pred_h3_%s.txt' % resolution, 'w') as g:
        for line in f:
            trip = json.loads(line)
            cells = []
            for c in trip['geometry']['coordinates']:
                # geo_to_h3(lat, lng, resolution)
                h = h3.geo_to_h3(c[1], c[0], resolution)
                cells.append(h)
            g.write('%s,%d,%d,%s\n' % (trip['properties']['VehicleNo'],
                datetime.fromtimestamp(trip['properties']['timestamps'][0]).hour,
                trip['properties']['duration'],' '.join(cells)))


