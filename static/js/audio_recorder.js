/**
 * Universal In-Browser 16-bit PCM WAV Audio Recorder
 * Generates uncompressed Microsoft WAV format with 100% cross-browser fidelity.
 * Project Exhibition – I (DSN2098) • Group-52
 */

class WavAudioRecorder {
    constructor() {
        this.audioContext = null;
        this.mediaStream = null;
        this.processor = null;
        this.inputNode = null;
        this.leftChannelData = [];
        this.recordingLength = 0;
        this.sampleRate = 44100;
        this.isRecording = false;
    }

    async start() {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.sampleRate = this.audioContext.sampleRate;
        this.inputNode = this.audioContext.createMediaStreamSource(this.mediaStream);
        
        // 4096 buffer size
        this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);
        this.leftChannelData = [];
        this.recordingLength = 0;
        this.isRecording = true;

        this.processor.onaudioprocess = (e) => {
            if (!this.isRecording) return;
            const left = e.inputBuffer.getChannelData(0);
            this.leftChannelData.push(new Float32Array(left));
            this.recordingLength += left.length;
        };

        this.inputNode.connect(this.processor);
        this.processor.connect(this.audioContext.destination);
    }

    async stop() {
        this.isRecording = false;
        if (this.processor && this.inputNode) {
            this.inputNode.disconnect();
            this.processor.disconnect();
        }
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(t => t.stop());
        }
        if (this.audioContext && this.audioContext.state !== 'closed') {
            await this.audioContext.close();
        }

        const merged = new Float32Array(this.recordingLength);
        let offset = 0;
        for (let i = 0; i < this.leftChannelData.length; i++) {
            merged.set(this.leftChannelData[i], offset);
            offset += this.leftChannelData[i].length;
        }

        return this.encodeWAV(merged, this.sampleRate);
    }

    encodeWAV(samples, sampleRate) {
        const buffer = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(buffer);

        this.writeString(view, 0, 'RIFF');
        view.setUint32(4, 36 + samples.length * 2, true);
        this.writeString(view, 8, 'WAVE');

        this.writeString(view, 12, 'fmt ');
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);

        this.writeString(view, 36, 'data');
        view.setUint32(40, samples.length * 2, true);

        let index = 44;
        for (let i = 0; i < samples.length; i++) {
            let s = Math.max(-1, Math.min(1, samples[i]));
            s = s < 0 ? s * 0x8000 : s * 0x7FFF;
            view.setInt16(index, s, true);
            index += 2;
        }

        return new Blob([view], { type: 'audio/wav' });
    }

    writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }
}

window.WavAudioRecorder = WavAudioRecorder;
