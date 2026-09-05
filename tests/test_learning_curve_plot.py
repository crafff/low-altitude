"""Exported figure title must survive iteration over subplot titles."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from learning_curve_plot import INPUT_SCHEMA, SCOPE, TITLE, render


class LearningCurveTitleTests(unittest.TestCase):
    def test_default_and_custom_titles_survive_actual_six_panel_export(self):
        aggregate = dict(planned=2, completed=2, failed_timeout=0,
                         failed_route_exhausted=0, outside_corridor_flights=0,
                         flight_hours=1., risk={level: dict(unordered_pair_seconds=0.,
                         unordered_seconds_per_flight_hour=0.) for level in ('lowc', 'nmac')})
        for custom in (None, 'Fresh navigation lineage diagnostic'):
            with self.subTest(title=custom), tempfile.TemporaryDirectory(prefix='learning-title-') as directory:
                root = Path(directory)
                document = dict(schema=INPUT_SCHEMA, scope=SCOPE,
                    rows=[dict(completed_episodes=0, nr=aggregate, sample=aggregate)],
                    source_records=[dict(run='owned-fixture', path='owned-fixture.json', sha256='0'*64, line=1)])
                if custom is not None:
                    document['title'] = custom
                source = root/'input.json'
                source.write_text(json.dumps(document))
                metadata = render(source, root)
                expected = TITLE if custom is None else custom
                self.assertEqual(metadata['title'], expected)
                svg = ET.parse(root/'learning_curve.svg')
                texts = [''.join(node.itertext()) for node in svg.findall('.//{http://www.w3.org/2000/svg}text')]
                self.assertIn(expected, texts)
                self.assertTrue((root/'learning_curve.png').stat().st_size > 0)
                self.assertTrue((root/'learning_curve.pdf').stat().st_size > 0)


if __name__ == '__main__':
    unittest.main()
