var map;

function initialize(){
    // Interface to OpenStreetMap and Google Maps
    var osmUrl = 'http://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
    var osmAttrib = '&copy; <a href="http://openstreetmap.org/copyright">OpenStreetMap</a> contributors';
    var googleUrl = 'http://www.google.com/maps/vt?lyrs=s@189&gl=cn&x={x}&y={y}&z={z}';
    var googleAttrib = '&copy; Google';

    var osm = L.tileLayer(osmUrl, { maxZoom: 18, attribution: osmAttrib });
    var google = L.tileLayer(googleUrl, { maxZoom: 18, attribution: googleAttrib });

    // Actual map object with drawn items
    map = new L.Map('map', { center: new L.LatLng(63.43011, 10.39478), zoom: 15 });
    var drawnItems = L.featureGroup().addTo(map);

    // Control for choosing which layers to show
    L.control.layers({
        'OpenStreetMap': osm.addTo(map),
        'Google': google,
    }, {
        'Drawn areas': drawnItems,
    }, {
        position: 'bottomleft',
        collapsed: true,
    }).addTo(map);

    // Full draw control: allowing drawing and editing
    var drawFull = new L.Control.Draw({
        edit: {
            featureGroup: drawnItems,
        },
        draw: {
            polyline: false,
            marker: false,
            circlemarker: false,
        },
    }).addTo(map);

    new QWebChannel(qt.webChannelTransport, function (channel) {
        window.MainWindow = channel.objects.Main;

        map.on(L.Draw.Event.CREATED, function (event) {
            drawnItems.addLayer(event.layer);
            if (typeof MainWindow != 'undefined') {
                MainWindow.update_objects(JSON.stringify(drawnItems.toGeoJSON()));
            }
        });

        map.on(L.Draw.Event.DELETED, function (event) {
            if (typeof MainWindow != 'undefined') {
                MainWindow.update_objects(JSON.stringify(drawnItems.toGeoJSON()));
            }
        });
    });
}
