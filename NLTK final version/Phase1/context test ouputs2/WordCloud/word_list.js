var item_count = 7;
var word_per_item = 10;
var width = 1200,
    height = 1200,
	fontSize = 12;

			var wordList = [
	{text: "robot",topic:0,sentiment:2,frequency:14.000,fitVal:0.0194},
	{text: "kit",topic:0,sentiment:2,frequency:9.000,fitVal:0.0194},
	{text: "robotics",topic:0,sentiment:2,frequency:7.000,fitVal:0.0194},
	{text: "competition",topic:0,sentiment:2,frequency:6.000,fitVal:0.0194},
	{text: "build",topic:0,sentiment:2,frequency:6.000,fitVal:0.0194},
	{text: "level",topic:0,sentiment:2,frequency:5.000,fitVal:0.0194},
	{text: "custom",topic:0,sentiment:2,frequency:5.000,fitVal:0.0194},
	{text: "season",topic:0,sentiment:2,frequency:4.000,fitVal:0.0194},
	{text: "school",topic:0,sentiment:2,frequency:4.000,fitVal:0.0194},
	{text: "team",topic:0,sentiment:2,frequency:4.000,fitVal:0.0194},
	{text: "time",topic:1,sentiment:2,frequency:6.000,fitVal:0.0136},
	{text: "idea",topic:1,sentiment:0,frequency:5.000,fitVal:0.0136},
	{text: "cute",topic:1,sentiment:1,frequency:5.000,fitVal:0.0136},
	{text: "create",topic:1,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "design",topic:1,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "sound",topic:1,sentiment:1,frequency:4.000,fitVal:0.0136},
	{text: "creating",topic:1,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "color",topic:1,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "robots",topic:1,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "beam",topic:1,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "art",topic:2,sentiment:2,frequency:5.000,fitVal:0.0136},
	{text: "rack",topic:2,sentiment:-1,frequency:5.000,fitVal:0.0136},
	{text: "waste",topic:2,sentiment:-1,frequency:4.000,fitVal:0.0136},
	{text: "pallet",topic:2,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "construction",topic:2,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "pallets",topic:2,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "create",topic:2,sentiment:2,frequency:4.000,fitVal:0.0136},
	{text: "recycled",topic:2,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "pieces",topic:2,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "recycle",topic:2,sentiment:2,frequency:3.000,fitVal:0.0136},
	{text: "size",topic:3,sentiment:2,frequency:13.000,fitVal:0.0083},
	{text: "barefoot",topic:3,sentiment:2,frequency:10.000,fitVal:0.0083},
	{text: "running",topic:3,sentiment:2,frequency:8.000,fitVal:0.0083},
	{text: "women's",topic:3,sentiment:2,frequency:7.000,fitVal:0.0083},
	{text: "men's",topic:3,sentiment:2,frequency:6.000,fitVal:0.0083},
	{text: "foot",topic:3,sentiment:2,frequency:5.000,fitVal:0.0083},
	{text: "extra",topic:3,sentiment:2,frequency:5.000,fitVal:0.0083},
	{text: "view",topic:3,sentiment:0,frequency:4.000,fitVal:0.0083},
	{text: "barepadz",topic:3,sentiment:2,frequency:4.000,fitVal:0.0083},
	{text: "footwear",topic:3,sentiment:2,frequency:4.000,fitVal:0.0083},
	{text: "sandals",topic:4,sentiment:2,frequency:19.000,fitVal:0.0083},
	{text: "size",topic:4,sentiment:2,frequency:11.000,fitVal:0.0083},
	{text: "laadi",topic:4,sentiment:2,frequency:8.000,fitVal:0.0083},
	{text: "collection",topic:4,sentiment:2,frequency:7.000,fitVal:0.0083},
	{text: "color",topic:4,sentiment:2,frequency:7.000,fitVal:0.0083},
	{text: "artisans",topic:4,sentiment:2,frequency:6.000,fitVal:0.0083},
	{text: "artisan",topic:4,sentiment:2,frequency:6.000,fitVal:0.0083},
	{text: "pair",topic:4,sentiment:2,frequency:5.000,fitVal:0.0083},
	{text: "fabric",topic:4,sentiment:2,frequency:5.000,fitVal:0.0083},
	{text: "sole",topic:4,sentiment:2,frequency:5.000,fitVal:0.0083},
	{text: "bibli",topic:5,sentiment:2,frequency:21.000,fitVal:0.0060},
	{text: "robot",topic:5,sentiment:2,frequency:9.000,fitVal:0.0060},
	{text: "longmont",topic:5,sentiment:2,frequency:6.000,fitVal:0.0060},
	{text: "library",topic:5,sentiment:2,frequency:5.000,fitVal:0.0060},
	{text: "fun",topic:5,sentiment:1,frequency:5.000,fitVal:0.0060},
	{text: "play",topic:5,sentiment:2,frequency:5.000,fitVal:0.0060},
	{text: "sound",topic:5,sentiment:1,frequency:5.000,fitVal:0.0060},
	{text: "artificial",topic:5,sentiment:-1,frequency:4.000,fitVal:0.0060},
	{text: "baby",topic:5,sentiment:0,frequency:4.000,fitVal:0.0060},
	{text: "designed",topic:5,sentiment:2,frequency:4.000,fitVal:0.0060},
	{text: "ringo",topic:6,sentiment:2,frequency:37.000,fitVal:0.0060},
	{text: "robot",topic:6,sentiment:2,frequency:17.000,fitVal:0.0060},
	{text: "code",topic:6,sentiment:2,frequency:13.000,fitVal:0.0060},
	{text: "light",topic:6,sentiment:1,frequency:12.000,fitVal:0.0060},
	{text: "arduino",topic:6,sentiment:2,frequency:11.000,fitVal:0.0060},
	{text: "sensors",topic:6,sentiment:2,frequency:9.000,fitVal:0.0060},
	{text: "lights",topic:6,sentiment:2,frequency:8.000,fitVal:0.0060},
	{text: "control",topic:6,sentiment:2,frequency:7.000,fitVal:0.0060},
	{text: "simple",topic:6,sentiment:1,frequency:7.000,fitVal:0.0060},
	{text: "axis",topic:6,sentiment:2,frequency:7.000,fitVal:0.0060},
];