using GraphicsRendererUI;

namespace GraphicsRendererUI.Models
{
    public class Light : ObservableObject, ISceneItem
    {
        private double _positionX = 0.0;
        private double _positionY = 5.0;
        private double _positionZ = 5.0;
        private double _colorR = 1.0;
        private double _colorG = 1.0;
        private double _colorB = 1.0;
        private double _luminosity = 5.0;
        private int _index = -1;
        private string _name = "";

        public string Name
        {
            get => _name;
            set => SetProperty(ref _name, value);
        }

        public double PositionX
        {
            get => _positionX;
            set => SetProperty(ref _positionX, value);
        }

        public double PositionY
        {
            get => _positionY;
            set => SetProperty(ref _positionY, value);
        }

        public double PositionZ
        {
            get => _positionZ;
            set => SetProperty(ref _positionZ, value);
        }

        public double ColorR
        {
            get => _colorR;
            set => SetProperty(ref _colorR, value);
        }

        public double ColorG
        {
            get => _colorG;
            set => SetProperty(ref _colorG, value);
        }

        public double ColorB
        {
            get => _colorB;
            set => SetProperty(ref _colorB, value);
        }

        public double Luminosity
        {
            get => _luminosity;
            set => SetProperty(ref _luminosity, value);
        }

        public int Index
        {
            get => _index;
            set => SetProperty(ref _index, value);
        }
    }
}
